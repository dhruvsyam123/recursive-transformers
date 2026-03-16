"""Core looped transformer model for Karatsuba multiplication.

Architecture:
- Single shared transformer block, looped via jax.lax.scan
- Pre-LayerNorm (more stable than post-LN)
- RMSNorm (more stable and efficient than LayerNorm for small models)
- Multi-head self-attention with causal masking (autoregressive)
- Timestep embedding (tells the model which loop iteration it is on)
- Support for fixed loop count and adaptive computation time (ACT)
- Output projection head for next-token prediction

Designed to be small (1-5M parameters with d=128-256, 4-8 heads). The key
innovation is the looped architecture, not model size.

Written in JAX + Equinox.
"""

from __future__ import annotations

from dataclasses import field as dataclass_field
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Bool

from src.model.position_encoding import (
    HierarchicalPositionEncoding,
    SinusoidalPositionEncoding,
    PositionCoupling,
    LearnablePositionEncoding,
)
from src.model.halting import (
    HaltingMechanism,
    adaptive_computation_time,
    fixed_loop,
    ponder_cost,
    ACTState,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TransformerConfig(eqx.Module):
    """Configuration dataclass for the looped transformer.

    All fields are static (not trainable) since they define architecture.
    """
    # Core dimensions
    d_model: int = eqx.field(static=True, default=256)
    n_heads: int = eqx.field(static=True, default=8)
    d_ff: int = eqx.field(static=True, default=512)       # FFN hidden dim (typically 2-4x d_model)
    n_shared_layers: int = eqx.field(static=True, default=1)  # Layers within one block (1-2)

    # Vocabulary and sequence
    vocab_size: int = eqx.field(static=True, default=16)   # Binary: {0, 1, special tokens}
    max_seq_len: int = eqx.field(static=True, default=4096)

    # Looping
    max_loop_iterations: int = eqx.field(static=True, default=32)
    use_act: bool = eqx.field(static=True, default=False)  # Adaptive computation time
    act_threshold: float = eqx.field(static=True, default=0.999)

    # Position encoding
    position_encoding_type: str = eqx.field(static=True, default="hierarchical")
    # Options: "hierarchical", "sinusoidal", "coupled", "learned"

    # Hierarchical position encoding parameters
    max_bit_significance: int = eqx.field(static=True, default=256)
    max_recursion_depth: int = eqx.field(static=True, default=16)
    max_sub_problem_id: int = eqx.field(static=True, default=64)
    num_step_types: int = eqx.field(static=True, default=7)

    # Regularization
    dropout_rate: float = eqx.field(static=True, default=0.0)

    # Attention
    use_causal_mask: bool = eqx.field(static=True, default=True)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    More stable and slightly faster than standard LayerNorm since it does
    not compute or subtract the mean. Preferred for small transformer models.

    Formula: x_norm = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    weight: Float[Array, " d_model"]
    eps: float = eqx.field(static=True)

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.weight = jnp.ones(d_model)
        self.eps = eps

    def __call__(self, x: Float[Array, " d_model"]) -> Float[Array, " d_model"]:
        """Normalize a single vector.

        Args:
            x: Input vector, shape (d_model,).

        Returns:
            Normalized vector, shape (d_model,).
        """
        rms = jnp.sqrt(jnp.mean(x ** 2) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Feed-forward network (SwiGLU variant)
# ---------------------------------------------------------------------------

class FeedForward(eqx.Module):
    """Feed-forward network with SwiGLU activation (Shazeer, 2020).

    SwiGLU: FFN(x) = (W1(x) * SiLU(W_gate(x))) @ W2
    Used in modern transformers (LLaMA, PaLM) for better performance.
    """
    w_gate: eqx.nn.Linear
    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear

    def __init__(self, d_model: int, d_ff: int, *, key: jax.random.PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w_gate = eqx.nn.Linear(d_model, d_ff, key=k1)
        self.w_up = eqx.nn.Linear(d_model, d_ff, key=k2)
        self.w_down = eqx.nn.Linear(d_ff, d_model, key=k3)

    def __call__(self, x: Float[Array, " d_model"]) -> Float[Array, " d_model"]:
        """Apply SwiGLU FFN to a single token.

        Args:
            x: Input, shape (d_model,).

        Returns:
            Output, shape (d_model,).
        """
        gate = jax.nn.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(eqx.Module):
    """Multi-head self-attention with support for causal masking.

    Uses standard scaled dot-product attention. Separate Q, K, V projections
    and output projection.
    """
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    w_q: eqx.nn.Linear
    w_k: eqx.nn.Linear
    w_v: eqx.nn.Linear
    w_o: eqx.nn.Linear

    def __init__(self, d_model: int, n_heads: int, *, key: jax.random.PRNGKey):
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.w_q = eqx.nn.Linear(d_model, d_model, key=k1)
        self.w_k = eqx.nn.Linear(d_model, d_model, key=k2)
        self.w_v = eqx.nn.Linear(d_model, d_model, key=k3)
        self.w_o = eqx.nn.Linear(d_model, d_model, key=k4)

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
    ) -> Float[Array, "seq_len d_model"]:
        """Apply multi-head self-attention.

        Args:
            x: Input sequence, shape (seq_len, d_model).
            mask: Boolean attention mask, shape (seq_len, seq_len).
                  True = attend, False = mask out.

        Returns:
            Output, shape (seq_len, d_model).
        """
        seq_len = x.shape[0]

        # Project Q, K, V
        q = jax.vmap(self.w_q)(x)  # (seq_len, d_model)
        k = jax.vmap(self.w_k)(x)
        v = jax.vmap(self.w_v)(x)

        # Reshape to (n_heads, seq_len, d_head)
        q = q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = jnp.sqrt(self.d_head).astype(x.dtype)
        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) / scale
        # attn_weights: (n_heads, seq_len, seq_len)

        # Apply mask
        if mask is not None:
            # Broadcast mask across heads: (1, seq_len, seq_len)
            attn_weights = jnp.where(
                mask[None, :, :],
                attn_weights,
                jnp.finfo(attn_weights.dtype).min,
            )

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Weighted sum of values
        attn_output = jnp.matmul(attn_weights, v)
        # attn_output: (n_heads, seq_len, d_head)

        # Reshape back to (seq_len, d_model)
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # Output projection
        output = jax.vmap(self.w_o)(attn_output)
        return output


# ---------------------------------------------------------------------------
# Single looped transformer block
# ---------------------------------------------------------------------------

class LoopedTransformerBlock(eqx.Module):
    """A single transformer block that is applied repeatedly (looped).

    Architecture: Pre-LayerNorm with RMSNorm.
        x = x + MHSA(RMSNorm(x + timestep_emb))
        x = x + FFN(RMSNorm(x))

    The timestep embedding is added before the first norm, conditioning the
    block on which loop iteration it is processing. This is critical for the
    model to know its current recursion depth.

    Following Saunshi et al. (ICLR 2025), the timestep embedding enables
    "latent thought" simulation equivalent to explicit chain-of-thought.
    """
    attention: MultiHeadSelfAttention
    ffn: FeedForward
    norm1: RMSNorm
    norm2: RMSNorm
    timestep_embed: eqx.nn.Embedding
    max_timesteps: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_timesteps: int = 64,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, key=k1)
        self.ffn = FeedForward(d_model, d_ff, key=k2)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.timestep_embed = eqx.nn.Embedding(
            num_embeddings=max_timesteps, embedding_size=d_model, key=k3
        )
        self.max_timesteps = max_timesteps

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        timestep: Int[Array, ""],
        mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
    ) -> Float[Array, "seq_len d_model"]:
        """Apply one loop iteration of the transformer block.

        Args:
            x: Input sequence, shape (seq_len, d_model).
            timestep: Scalar integer indicating the current loop iteration.
            mask: Optional attention mask, shape (seq_len, seq_len).

        Returns:
            Updated sequence, shape (seq_len, d_model).
        """
        # Add timestep embedding (broadcast across sequence)
        t_emb = self.timestep_embed(timestep)  # (d_model,)
        x_conditioned = x + t_emb[None, :]  # (seq_len, d_model)

        # Pre-norm self-attention
        normed = jax.vmap(self.norm1)(x_conditioned)
        x = x + self.attention(normed, mask=mask)

        # Pre-norm feed-forward
        normed = jax.vmap(self.norm2)(x)
        x = x + jax.vmap(self.ffn)(normed)

        return x


# ---------------------------------------------------------------------------
# Multi-layer shared block (for n_shared_layers > 1)
# ---------------------------------------------------------------------------

class SharedTransformerLayers(eqx.Module):
    """Multiple transformer layers that together form one "shared block".

    If n_shared_layers=1, this is just a single LoopedTransformerBlock.
    If n_shared_layers=2, two different blocks are applied sequentially
    within each loop iteration.

    The loop repeats the entire stack of shared layers.
    """
    layers: list
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_timesteps: int = 64,
        *,
        key: jax.random.PRNGKey,
    ):
        self.n_layers = n_layers
        keys = jax.random.split(key, n_layers)
        self.layers = [
            LoopedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_timesteps=max_timesteps,
                key=k,
            )
            for k in keys
        ]

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        timestep: Int[Array, ""],
        mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
    ) -> Float[Array, "seq_len d_model"]:
        """Apply all shared layers sequentially.

        Args:
            x: Input, shape (seq_len, d_model).
            timestep: Current loop iteration index.
            mask: Optional attention mask.

        Returns:
            Output, shape (seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, timestep, mask=mask)
        return x


# ---------------------------------------------------------------------------
# Full looped transformer model
# ---------------------------------------------------------------------------

class LoopedTransformer(eqx.Module):
    """Complete looped transformer for Karatsuba multiplication.

    Architecture overview:
    1. Token embedding + position encoding
    2. Shared transformer block(s) looped N times (via jax.lax.scan)
    3. Final RMSNorm
    4. Output projection to vocabulary logits

    Supports:
    - Fixed loop count (simpler, faster, recommended initially)
    - Adaptive computation time (ACT) with learned halting
    - Multiple position encoding schemes (hierarchical, sinusoidal,
      coupled, learned)

    Small by design: 1-5M parameters with d=128-256, 4-8 heads.
    """
    config: TransformerConfig

    # Token embedding
    token_embed: eqx.nn.Embedding

    # Position encoding (one of several types)
    pos_enc_type: str = eqx.field(static=True)
    hierarchical_pos: Optional[HierarchicalPositionEncoding]
    sinusoidal_pos: Optional[SinusoidalPositionEncoding]
    coupled_pos: Optional[PositionCoupling]
    learned_pos: Optional[LearnablePositionEncoding]

    # Shared transformer block(s)
    shared_block: SharedTransformerLayers

    # Final normalization and output head
    final_norm: RMSNorm
    output_head: eqx.nn.Linear

    # Adaptive computation time (optional)
    halting: Optional[HaltingMechanism]

    def __init__(self, config: TransformerConfig, *, key: jax.random.PRNGKey):
        """Initialize the looped transformer.

        Args:
            config: TransformerConfig with all architecture hyperparameters.
            key: PRNG key for weight initialization.
        """
        self.config = config
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # Token embedding
        self.token_embed = eqx.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_size=config.d_model,
            key=k1,
        )

        # Position encoding
        self.pos_enc_type = config.position_encoding_type

        if config.position_encoding_type == "hierarchical":
            self.hierarchical_pos = HierarchicalPositionEncoding(
                d_model=config.d_model,
                max_bit_significance=config.max_bit_significance,
                max_recursion_depth=config.max_recursion_depth,
                max_sub_problem_id=config.max_sub_problem_id,
                num_step_types=config.num_step_types,
                key=k2,
            )
            self.sinusoidal_pos = None
            self.coupled_pos = None
            self.learned_pos = None
        elif config.position_encoding_type == "sinusoidal":
            self.sinusoidal_pos = SinusoidalPositionEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
            )
            self.hierarchical_pos = None
            self.coupled_pos = None
            self.learned_pos = None
        elif config.position_encoding_type == "coupled":
            self.coupled_pos = PositionCoupling(
                d_model=config.d_model,
                max_coupled_positions=config.max_seq_len,
            )
            self.hierarchical_pos = None
            self.sinusoidal_pos = None
            self.learned_pos = None
        elif config.position_encoding_type == "learned":
            self.learned_pos = LearnablePositionEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
                key=k2,
            )
            self.hierarchical_pos = None
            self.sinusoidal_pos = None
            self.coupled_pos = None
        else:
            raise ValueError(
                f"Unknown position encoding type: {config.position_encoding_type}"
            )

        # Shared transformer block(s)
        self.shared_block = SharedTransformerLayers(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_shared_layers,
            max_timesteps=config.max_loop_iterations,
            key=k3,
        )

        # Final norm and output projection
        self.final_norm = RMSNorm(config.d_model)
        self.output_head = eqx.nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            key=k4,
        )

        # ACT halting mechanism (optional)
        if config.use_act:
            self.halting = HaltingMechanism(
                d_model=config.d_model,
                threshold=config.act_threshold,
                key=k5,
            )
        else:
            self.halting = None

    def embed_tokens(
        self, tokens: Int[Array, " seq_len"]
    ) -> Float[Array, "seq_len d_model"]:
        """Embed token IDs into d_model-dimensional vectors.

        Args:
            tokens: Token IDs, shape (seq_len,).

        Returns:
            Token embeddings, shape (seq_len, d_model).
        """
        return jax.vmap(self.token_embed)(tokens)

    def add_position_encoding_hierarchical(
        self,
        x: Float[Array, "seq_len d_model"],
        bit_significance: Int[Array, " seq_len"],
        recursion_depth: Int[Array, " seq_len"],
        sub_problem_id: Int[Array, " seq_len"],
        step_type: Int[Array, " seq_len"],
    ) -> Float[Array, "seq_len d_model"]:
        """Add hierarchical position encoding.

        Args:
            x: Token embeddings, shape (seq_len, d_model).
            bit_significance: Bit position indices.
            recursion_depth: Recursion depth indices.
            sub_problem_id: Sub-problem ID indices.
            step_type: Step type indices.

        Returns:
            Embeddings with position information, shape (seq_len, d_model).
        """
        pos_enc = self.hierarchical_pos(
            bit_significance, recursion_depth, sub_problem_id, step_type
        )
        return x + pos_enc

    def add_position_encoding_simple(
        self,
        x: Float[Array, "seq_len d_model"],
        positions: Int[Array, " seq_len"],
    ) -> Float[Array, "seq_len d_model"]:
        """Add sinusoidal, coupled, or learned position encoding.

        Args:
            x: Token embeddings, shape (seq_len, d_model).
            positions: Position indices, shape (seq_len,).

        Returns:
            Embeddings with position information, shape (seq_len, d_model).
        """
        if self.pos_enc_type == "sinusoidal":
            return x + self.sinusoidal_pos(positions)
        elif self.pos_enc_type == "coupled":
            return x + self.coupled_pos(positions)
        elif self.pos_enc_type == "learned":
            return x + self.learned_pos(positions)
        else:
            raise ValueError(f"Expected simple position encoding, got {self.pos_enc_type}")

    def make_causal_mask(
        self, seq_len: int
    ) -> Bool[Array, "seq_len seq_len"]:
        """Create a causal (lower-triangular) attention mask.

        Args:
            seq_len: Sequence length.

        Returns:
            Boolean mask where True = attend, False = mask out.
            Shape: (seq_len, seq_len).
        """
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    def forward_fixed_loops(
        self,
        x: Float[Array, "seq_len d_model"],
        n_loops: int,
        mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
    ) -> Float[Array, "seq_len d_model"]:
        """Forward pass with a fixed number of loop iterations.

        Args:
            x: Input embeddings (tokens + positions), shape (seq_len, d_model).
            n_loops: Number of loop iterations.
            mask: Optional attention mask.

        Returns:
            Output hidden states, shape (seq_len, d_model).
        """
        def block_fn(hidden, timestep):
            return self.shared_block(hidden, timestep, mask=mask)

        return fixed_loop(block_fn, x, n_loops)

    def forward_act(
        self,
        x: Float[Array, "seq_len d_model"],
        max_loops: int,
        mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
    ) -> tuple[Float[Array, "seq_len d_model"], ACTState]:
        """Forward pass with adaptive computation time.

        Args:
            x: Input embeddings, shape (seq_len, d_model).
            max_loops: Maximum number of loop iterations.
            mask: Optional attention mask.

        Returns:
            Tuple of (output hidden states, ACT state for ponder cost).
        """
        def block_fn(hidden, timestep):
            return self.shared_block(hidden, timestep, mask=mask)

        return adaptive_computation_time(
            block_fn, self.halting, x, max_loops
        )

    def __call__(
        self,
        tokens: Int[Array, " seq_len"],
        positions: Int[Array, " seq_len"],
        n_loops: Optional[int] = None,
        bit_significance: Optional[Int[Array, " seq_len"]] = None,
        recursion_depth: Optional[Int[Array, " seq_len"]] = None,
        sub_problem_id: Optional[Int[Array, " seq_len"]] = None,
        step_type: Optional[Int[Array, " seq_len"]] = None,
    ) -> Float[Array, "seq_len vocab_size"] | tuple[Float[Array, "seq_len vocab_size"], ACTState]:
        """Full forward pass: embed -> loop -> project.

        For hierarchical position encoding, pass the 4 structural components.
        For other encodings, only `positions` is used.

        Args:
            tokens: Token IDs, shape (seq_len,).
            positions: Position indices, shape (seq_len,).
                For hierarchical encoding, this is ignored (use the 4 components).
                For other encodings, this is the position ID.
            n_loops: Number of loop iterations. If None, uses config.max_loop_iterations.
            bit_significance: (Hierarchical only) Bit position indices.
            recursion_depth: (Hierarchical only) Recursion depth indices.
            sub_problem_id: (Hierarchical only) Sub-problem ID indices.
            step_type: (Hierarchical only) Step type indices.

        Returns:
            If use_act=False: logits of shape (seq_len, vocab_size).
            If use_act=True: tuple of (logits, ACTState).
        """
        seq_len = tokens.shape[0]
        n_loops = n_loops if n_loops is not None else self.config.max_loop_iterations

        # 1. Token embedding
        x = self.embed_tokens(tokens)

        # 2. Add position encoding
        if self.pos_enc_type == "hierarchical":
            assert bit_significance is not None, (
                "Hierarchical encoding requires bit_significance"
            )
            assert recursion_depth is not None, (
                "Hierarchical encoding requires recursion_depth"
            )
            assert sub_problem_id is not None, (
                "Hierarchical encoding requires sub_problem_id"
            )
            assert step_type is not None, (
                "Hierarchical encoding requires step_type"
            )
            x = self.add_position_encoding_hierarchical(
                x, bit_significance, recursion_depth, sub_problem_id, step_type
            )
        else:
            x = self.add_position_encoding_simple(x, positions)

        # 3. Create attention mask
        mask = self.make_causal_mask(seq_len) if self.config.use_causal_mask else None

        # 4. Apply looped transformer block(s)
        act_state = None
        if self.config.use_act and self.halting is not None:
            x, act_state = self.forward_act(x, n_loops, mask=mask)
        else:
            x = self.forward_fixed_loops(x, n_loops, mask=mask)

        # 5. Final normalization
        x = jax.vmap(self.final_norm)(x)

        # 6. Project to vocabulary logits
        logits = jax.vmap(self.output_head)(x)  # (seq_len, vocab_size)

        if act_state is not None:
            return logits, act_state
        return logits


# ---------------------------------------------------------------------------
# Helper: compute loss
# ---------------------------------------------------------------------------

def compute_loss(
    model: LoopedTransformer,
    tokens: Int[Array, " seq_len"],
    targets: Int[Array, " seq_len"],
    positions: Int[Array, " seq_len"],
    n_loops: Optional[int] = None,
    bit_significance: Optional[Int[Array, " seq_len"]] = None,
    recursion_depth: Optional[Int[Array, " seq_len"]] = None,
    sub_problem_id: Optional[Int[Array, " seq_len"]] = None,
    step_type: Optional[Int[Array, " seq_len"]] = None,
    ponder_weight: float = 0.01,
    loss_mask: Optional[Bool[Array, " seq_len"]] = None,
) -> Float[Array, ""]:
    """Compute cross-entropy loss with optional ponder cost.

    Args:
        model: LoopedTransformer instance.
        tokens: Input token IDs, shape (seq_len,).
        targets: Target token IDs for next-token prediction, shape (seq_len,).
        positions: Position indices.
        n_loops: Number of loop iterations.
        bit_significance, recursion_depth, sub_problem_id, step_type:
            Hierarchical position components (optional).
        ponder_weight: Weight for ACT ponder cost regularization.
        loss_mask: Optional boolean mask for which tokens to compute loss on.
            Shape (seq_len,). True = include in loss, False = ignore.

    Returns:
        Scalar loss value.
    """
    output = model(
        tokens, positions, n_loops,
        bit_significance=bit_significance,
        recursion_depth=recursion_depth,
        sub_problem_id=sub_problem_id,
        step_type=step_type,
    )

    if isinstance(output, tuple):
        logits, act_state = output
    else:
        logits = output
        act_state = None

    # Cross-entropy loss
    # logits: (seq_len, vocab_size), targets: (seq_len,)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Gather the log-prob of the correct target for each position
    target_log_probs = log_probs[jnp.arange(targets.shape[0]), targets]

    if loss_mask is not None:
        # Only compute loss on masked positions
        loss = -jnp.sum(target_log_probs * loss_mask) / jnp.maximum(
            jnp.sum(loss_mask), 1.0
        )
    else:
        loss = -jnp.mean(target_log_probs)

    # Add ponder cost if using ACT
    if act_state is not None:
        loss = loss + ponder_weight * ponder_cost(act_state)

    return loss


# ---------------------------------------------------------------------------
# Helper: count parameters
# ---------------------------------------------------------------------------

def count_parameters(model: eqx.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: Any Equinox module.

    Returns:
        Total parameter count.
    """
    params = eqx.filter(model, eqx.is_array)
    return sum(x.size for x in jax.tree.leaves(params))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_model(
    config: Optional[TransformerConfig] = None,
    *,
    key: jax.random.PRNGKey,
    **config_overrides,
) -> LoopedTransformer:
    """Create a LoopedTransformer with the given configuration.

    Args:
        config: TransformerConfig. If None, uses defaults.
        key: PRNG key.
        **config_overrides: Override specific config fields.

    Returns:
        Initialized LoopedTransformer.

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> model = create_model(key=key, d_model=128, n_heads=4)
        >>> print(f"Parameters: {count_parameters(model):,}")
    """
    if config is None:
        config = TransformerConfig(**config_overrides)
    model = LoopedTransformer(config, key=key)
    return model
