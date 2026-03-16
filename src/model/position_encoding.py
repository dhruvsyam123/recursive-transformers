"""Hierarchical position encodings for the Karatsuba looped transformer.

Implements four position encoding strategies:
1. Standard sinusoidal (baseline)
2. Hierarchical 4-component encoding (bit_significance, recursion_depth,
   sub_problem_id, step_type) -- the core innovation for Karatsuba structure
3. Position coupling (tokens with same structural role share position IDs)
4. Learnable position embeddings (alternative)

All implementations use JAX + Equinox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int


# ---------------------------------------------------------------------------
# 1. Standard sinusoidal position encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionEncoding(eqx.Module):
    """Standard sinusoidal position encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Used as a baseline and as a building block for hierarchical encodings.
    """
    d_model: int = eqx.field(static=True)
    max_len: int = eqx.field(static=True)
    # Pre-computed encoding table: (max_len, d_model)
    _encoding_table: Float[Array, "max_len d_model"]

    def __init__(self, d_model: int, max_len: int = 4096):
        self.d_model = d_model
        self.max_len = max_len

        positions = jnp.arange(max_len, dtype=jnp.float32)[:, None]  # (max_len, 1)
        dim_indices = jnp.arange(0, d_model, 2, dtype=jnp.float32)  # (d_model/2,)
        # Compute frequencies: 1 / 10000^(2i/d)
        freqs = jnp.exp(-dim_indices * (math.log(10000.0) / d_model))  # (d_model/2,)
        # Compute angles
        angles = positions * freqs  # (max_len, d_model/2)
        # Interleave sin and cos
        sin_enc = jnp.sin(angles)
        cos_enc = jnp.cos(angles)
        # Stack: [sin_0, cos_0, sin_1, cos_1, ...]
        encoding = jnp.zeros((max_len, d_model))
        encoding = encoding.at[:, 0::2].set(sin_enc)
        encoding = encoding.at[:, 1::2].set(cos_enc)
        self._encoding_table = encoding

    def __call__(
        self, positions: Int[Array, " seq_len"]
    ) -> Float[Array, "seq_len d_model"]:
        """Look up sinusoidal encodings for given position indices.

        Args:
            positions: Integer position IDs, shape (seq_len,).
                       Values should be in [0, max_len).

        Returns:
            Position encodings, shape (seq_len, d_model).
        """
        return self._encoding_table[positions]

    def encode_continuous(
        self, positions: Float[Array, " seq_len"]
    ) -> Float[Array, "seq_len d_model"]:
        """Encode continuous (non-integer) position values.

        Useful for interpolation experiments.
        """
        positions = positions[:, None]  # (seq_len, 1)
        dim_indices = jnp.arange(0, self.d_model, 2, dtype=jnp.float32)
        freqs = jnp.exp(-dim_indices * (math.log(10000.0) / self.d_model))
        angles = positions * freqs  # (seq_len, d_model/2)
        encoding = jnp.zeros((positions.shape[0], self.d_model))
        encoding = encoding.at[:, 0::2].set(jnp.sin(angles))
        encoding = encoding.at[:, 1::2].set(jnp.cos(angles))
        return encoding


# ---------------------------------------------------------------------------
# 2. Hierarchical position encoding
# ---------------------------------------------------------------------------

class HierarchicalPositionEncoding(eqx.Module):
    """Hierarchical position encoding with 4 structural components.

    Each token in the Karatsuba scratchpad is associated with a 4-tuple:
        (bit_significance, recursion_depth, sub_problem_id, step_type)

    Each component is encoded independently (either via sinusoidal or learned
    embeddings) and the results are summed/concatenated to form the final
    position encoding.

    Components:
        - bit_significance: which bit position (0=LSB, n-1=MSB). Tells the
          model which half a bit belongs to after Karatsuba splitting.
        - recursion_depth: depth in the recursion tree (0=top, log2(n)=base).
        - sub_problem_id: which of the 3 sub-problems (z0=0, z1=1, z2=2) at
          the current recursion level.
        - step_type: what operation (SPLIT=0, MULTIPLY=1, ADD=2, SUB=3,
          COMBINE=4, INPUT=5, OUTPUT=6).

    Supports two combination modes:
        - "sum": Each component is encoded into d_model dims and summed.
        - "concat": Each component is encoded into d_model/4 dims and concatenated.
    """
    d_model: int = eqx.field(static=True)
    combine_mode: str = eqx.field(static=True)
    max_bit_significance: int = eqx.field(static=True)
    max_recursion_depth: int = eqx.field(static=True)
    max_sub_problem_id: int = eqx.field(static=True)
    num_step_types: int = eqx.field(static=True)

    # Per-component encoders
    bit_sig_encoder: SinusoidalPositionEncoding
    depth_encoder: eqx.nn.Embedding
    sub_problem_encoder: eqx.nn.Embedding
    step_type_encoder: eqx.nn.Embedding

    # Optional learned mixing weights per component
    component_weights: Float[Array, "4"]

    def __init__(
        self,
        d_model: int,
        max_bit_significance: int = 256,
        max_recursion_depth: int = 16,
        max_sub_problem_id: int = 64,
        num_step_types: int = 7,
        combine_mode: str = "sum",
        *,
        key: jax.random.PRNGKey,
    ):
        """
        Args:
            d_model: Model dimension.
            max_bit_significance: Maximum number of bit positions to support.
            max_recursion_depth: Maximum recursion depth (log2 of max bit width).
            max_sub_problem_id: Maximum sub-problem ID (3^depth in the worst
                case, but we cap it for the embedding table).
            num_step_types: Number of distinct step types (SPLIT, MULTIPLY,
                ADD, SUB, COMBINE, INPUT, OUTPUT).
            combine_mode: "sum" or "concat".
            key: PRNG key for initializing learned embeddings.
        """
        self.d_model = d_model
        self.combine_mode = combine_mode
        self.max_bit_significance = max_bit_significance
        self.max_recursion_depth = max_recursion_depth
        self.max_sub_problem_id = max_sub_problem_id
        self.num_step_types = num_step_types

        k1, k2, k3, k4 = jax.random.split(key, 4)

        if combine_mode == "concat":
            assert d_model % 4 == 0, (
                f"d_model ({d_model}) must be divisible by 4 for concat mode"
            )
            component_dim = d_model // 4
        else:
            component_dim = d_model

        # Bit significance: sinusoidal (captures the periodic/hierarchical
        # structure of binary numbers naturally).
        self.bit_sig_encoder = SinusoidalPositionEncoding(
            d_model=component_dim, max_len=max_bit_significance
        )

        # Recursion depth: learned embedding (small vocabulary, discrete).
        self.depth_encoder = eqx.nn.Embedding(
            num_embeddings=max_recursion_depth, embedding_size=component_dim, key=k2
        )

        # Sub-problem ID: learned embedding.
        self.sub_problem_encoder = eqx.nn.Embedding(
            num_embeddings=max_sub_problem_id, embedding_size=component_dim, key=k3
        )

        # Step type: learned embedding.
        self.step_type_encoder = eqx.nn.Embedding(
            num_embeddings=num_step_types, embedding_size=component_dim, key=k4
        )

        # Learnable per-component scaling (initialized to uniform).
        self.component_weights = jnp.ones(4) * 0.25

    def __call__(
        self,
        bit_significance: Int[Array, " seq_len"],
        recursion_depth: Int[Array, " seq_len"],
        sub_problem_id: Int[Array, " seq_len"],
        step_type: Int[Array, " seq_len"],
    ) -> Float[Array, "seq_len d_model"]:
        """Compute hierarchical position encoding.

        Args:
            bit_significance: Bit position indices, shape (seq_len,).
            recursion_depth: Recursion depth indices, shape (seq_len,).
            sub_problem_id: Sub-problem indices, shape (seq_len,).
            step_type: Step type indices, shape (seq_len,).

        Returns:
            Position encodings, shape (seq_len, d_model).
        """
        enc_bit = self.bit_sig_encoder(bit_significance)      # (seq_len, comp_dim)
        enc_depth = jax.vmap(self.depth_encoder)(recursion_depth)  # (seq_len, comp_dim)
        enc_sub = jax.vmap(self.sub_problem_encoder)(sub_problem_id)
        enc_step = jax.vmap(self.step_type_encoder)(step_type)

        if self.combine_mode == "concat":
            # Concatenate along feature dimension
            return jnp.concatenate(
                [enc_bit, enc_depth, enc_sub, enc_step], axis=-1
            )
        else:
            # Weighted sum
            w = jax.nn.softmax(self.component_weights)
            return (
                w[0] * enc_bit
                + w[1] * enc_depth
                + w[2] * enc_sub
                + w[3] * enc_step
            )


# ---------------------------------------------------------------------------
# 3. Position coupling
# ---------------------------------------------------------------------------

class PositionCoupling(eqx.Module):
    """Position coupling: tokens with the same structural role share position IDs.

    Following Cho et al. (NeurIPS 2024), digits of the same significance in
    different numbers (input operands, intermediate results, output) receive the
    same position ID. This directly encodes the column-wise alignment that
    arithmetic requires.

    For Karatsuba, position coupling goes further: tokens representing the same
    bit significance within the same sub-problem at the same recursion level
    share position IDs. This encodes the recursive structure into attention
    patterns.

    This module maps (structural_role_tuple) -> coupled_position_id, then
    applies a base position encoding (sinusoidal or learned) to those IDs.
    """
    d_model: int = eqx.field(static=True)
    max_coupled_positions: int = eqx.field(static=True)
    base_encoding: SinusoidalPositionEncoding

    def __init__(self, d_model: int, max_coupled_positions: int = 4096):
        self.d_model = d_model
        self.max_coupled_positions = max_coupled_positions
        self.base_encoding = SinusoidalPositionEncoding(
            d_model=d_model, max_len=max_coupled_positions
        )

    def __call__(
        self, coupled_position_ids: Int[Array, " seq_len"]
    ) -> Float[Array, "seq_len d_model"]:
        """Encode pre-computed coupled position IDs.

        The caller is responsible for assigning coupled position IDs such that
        tokens with the same structural role get the same ID. This module
        simply encodes those IDs.

        Args:
            coupled_position_ids: Pre-assigned coupled position IDs,
                shape (seq_len,). Tokens sharing a structural role have
                the same ID.

        Returns:
            Position encodings, shape (seq_len, d_model).
        """
        return self.base_encoding(coupled_position_ids)

    @staticmethod
    def compute_coupled_ids_karatsuba(
        bit_significance: Int[Array, " seq_len"],
        recursion_depth: Int[Array, " seq_len"],
        sub_problem_id: Int[Array, " seq_len"],
        max_bit_sig: int = 256,
        max_depth: int = 16,
    ) -> Int[Array, " seq_len"]:
        """Compute coupled position IDs from Karatsuba structural info.

        Tokens with the same (bit_significance, recursion_depth, sub_problem_id)
        triple receive the same coupled position ID. The step_type is NOT used
        for coupling -- we want input digits, intermediate results, and output
        digits of the same significance to attend to each other.

        Args:
            bit_significance: Bit position, shape (seq_len,).
            recursion_depth: Recursion depth, shape (seq_len,).
            sub_problem_id: Sub-problem ID, shape (seq_len,).
            max_bit_sig: Maximum bit significance value (for hashing).
            max_depth: Maximum recursion depth (for hashing).

        Returns:
            Coupled position IDs, shape (seq_len,).
        """
        # Hash the triple into a single integer ID.
        # ID = bit_sig + max_bit_sig * (depth + max_depth * sub_problem_id)
        coupled_ids = (
            bit_significance
            + max_bit_sig * (recursion_depth + max_depth * sub_problem_id)
        )
        return coupled_ids

    @staticmethod
    def compute_coupled_ids_simple(
        bit_significance: Int[Array, " seq_len"],
    ) -> Int[Array, " seq_len"]:
        """Simple position coupling: only couple by bit significance.

        This is the simplest form of coupling from Cho et al. -- digits of the
        same column (ones, tens, hundreds, ...) share a position ID regardless
        of which number they belong to.

        Args:
            bit_significance: Bit position, shape (seq_len,).

        Returns:
            Coupled position IDs = bit_significance directly.
        """
        return bit_significance


# ---------------------------------------------------------------------------
# 4. Learnable position embeddings
# ---------------------------------------------------------------------------

class LearnablePositionEncoding(eqx.Module):
    """Fully learnable absolute position embeddings.

    Each position gets a learned d_model-dimensional vector. Simple but does
    not generalise to unseen lengths by default. Included as an ablation
    baseline.

    Can optionally support hierarchical structure by using separate learned
    embeddings for each structural component and summing them (similar to
    HierarchicalPositionEncoding but with all components learned).
    """
    d_model: int = eqx.field(static=True)
    max_len: int = eqx.field(static=True)
    embedding: eqx.nn.Embedding

    def __init__(self, d_model: int, max_len: int = 4096, *, key: jax.random.PRNGKey):
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = eqx.nn.Embedding(
            num_embeddings=max_len, embedding_size=d_model, key=key
        )

    def __call__(
        self, positions: Int[Array, " seq_len"]
    ) -> Float[Array, "seq_len d_model"]:
        """Look up learned position embeddings.

        Args:
            positions: Position indices, shape (seq_len,). Values in [0, max_len).

        Returns:
            Position embeddings, shape (seq_len, d_model).
        """
        return jax.vmap(self.embedding)(positions)


# ---------------------------------------------------------------------------
# Step type constants (for convenience)
# ---------------------------------------------------------------------------

STEP_SPLIT = 0
STEP_MULTIPLY = 1
STEP_ADD = 2
STEP_SUB = 3
STEP_COMBINE = 4
STEP_INPUT = 5
STEP_OUTPUT = 6
