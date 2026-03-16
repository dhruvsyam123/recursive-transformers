"""Adaptive Computation Time (ACT) halting mechanism.

Implements the pondering / halting mechanism from:
- Dehghani et al. (2018), "Universal Transformers"
- Graves (2016), "Adaptive Computation Time for Recurrent Neural Networks"
- Fan et al. (ICLR 2025), "Looped Transformers for Length Generalization"

The key idea: each token at each loop iteration produces a halting
probability. Once the cumulative halting probability exceeds a threshold,
that token stops being updated. A remainder distribution ensures the
halting weights sum to 1, enabling smooth gradient flow.

Compatible with:
- jax.lax.scan with masking (fixed max iterations, mask out halted tokens)
- jax.lax.while_loop (truly adaptive, but harder to compile efficiently)

All implementations use JAX + Equinox.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Bool


# ---------------------------------------------------------------------------
# ACT state container
# ---------------------------------------------------------------------------

class ACTState(NamedTuple):
    """State carried through the ACT loop.

    Attributes:
        halting_prob: Cumulative halting probability per token.
            Shape: (seq_len,). Starts at 0, monotonically increases to 1.
        remainders: Remainder probability assigned at the final step.
            Shape: (seq_len,). Non-zero only for tokens that just halted.
        n_updates: Number of updates each token has received so far.
            Shape: (seq_len,). Integer count.
        halted: Boolean mask indicating which tokens have halted.
            Shape: (seq_len,).
        accumulated_output: Weighted sum of hidden states across iterations.
            Shape: (seq_len, d_model). The final output is this value after
            all tokens have halted.
    """
    halting_prob: Float[Array, " seq_len"]
    remainders: Float[Array, " seq_len"]
    n_updates: Float[Array, " seq_len"]
    halted: Bool[Array, " seq_len"]
    accumulated_output: Float[Array, "seq_len d_model"]


# ---------------------------------------------------------------------------
# Halting mechanism module
# ---------------------------------------------------------------------------

class HaltingMechanism(eqx.Module):
    """Produces per-token halting probabilities from hidden states.

    A small MLP (or single linear layer) that takes the current hidden state
    and outputs a scalar halting probability in (0, 1) via sigmoid.
    """
    halt_linear: eqx.nn.Linear
    threshold: float = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        threshold: float = 1.0 - 1e-3,
        *,
        key: jax.random.PRNGKey,
    ):
        """
        Args:
            d_model: Hidden state dimension.
            threshold: Cumulative probability threshold for halting.
                Default 1 - epsilon (Graves, 2016).
            key: PRNG key.
        """
        self.halt_linear = eqx.nn.Linear(
            in_features=d_model, out_features=1, key=key
        )
        self.threshold = threshold

    def __call__(
        self, hidden_states: Float[Array, "seq_len d_model"]
    ) -> Float[Array, " seq_len"]:
        """Compute per-token halting probabilities.

        Args:
            hidden_states: Current hidden states, shape (seq_len, d_model).

        Returns:
            Halting probabilities in (0, 1), shape (seq_len,).
        """
        # Apply linear + sigmoid to each token independently
        logits = jax.vmap(self.halt_linear)(hidden_states)  # (seq_len, 1)
        probs = jax.nn.sigmoid(logits.squeeze(-1))          # (seq_len,)
        return probs


# ---------------------------------------------------------------------------
# ACT step function (for use inside jax.lax.scan)
# ---------------------------------------------------------------------------

def act_step(
    halting_mechanism: HaltingMechanism,
    hidden_states: Float[Array, "seq_len d_model"],
    act_state: ACTState,
) -> ACTState:
    """One step of Adaptive Computation Time.

    Given current hidden states and the ACT state, compute halting
    probabilities, determine which tokens halt, and accumulate weighted
    hidden states.

    Args:
        halting_mechanism: Module that produces halting probabilities.
        hidden_states: Current hidden states from the transformer block.
            Shape: (seq_len, d_model).
        act_state: Current ACT state.

    Returns:
        Updated ACT state.
    """
    seq_len = hidden_states.shape[0]

    # Compute raw halting probability for each token
    p = halting_mechanism(hidden_states)  # (seq_len,)

    # Tokens that have already halted should not be updated
    still_running = ~act_state.halted  # (seq_len,)

    # What would the cumulative probability be if we added p?
    new_halting_prob = act_state.halting_prob + p * still_running.astype(jnp.float32)

    # Determine which tokens halt at this step (cross the threshold)
    halts_now = (new_halting_prob >= halting_mechanism.threshold) & still_running

    # For tokens that halt now, the weight is the remainder (1 - previous cumulative)
    # For tokens still running, the weight is p
    # For tokens already halted, the weight is 0
    remainder = 1.0 - act_state.halting_prob
    weight = jnp.where(
        halts_now,
        remainder,  # Use remainder for newly halted tokens
        jnp.where(
            still_running,
            p,         # Use raw probability for still-running tokens
            0.0,       # Zero for already-halted tokens
        ),
    )

    # Update cumulative halting probability
    updated_halting_prob = jnp.where(
        halts_now,
        1.0,  # Clamp to 1.0 for halted tokens
        new_halting_prob,
    )

    # Update remainders
    updated_remainders = jnp.where(
        halts_now,
        remainder,
        act_state.remainders,
    )

    # Accumulate weighted hidden states
    updated_output = (
        act_state.accumulated_output
        + weight[:, None] * hidden_states
    )

    # Update halted mask
    updated_halted = act_state.halted | halts_now

    # Update step count
    updated_n_updates = act_state.n_updates + still_running.astype(jnp.float32)

    return ACTState(
        halting_prob=updated_halting_prob,
        remainders=updated_remainders,
        n_updates=updated_n_updates,
        halted=updated_halted,
        accumulated_output=updated_output,
    )


# ---------------------------------------------------------------------------
# Ponder cost (regularization loss)
# ---------------------------------------------------------------------------

def ponder_cost(act_state: ACTState) -> Float[Array, ""]:
    """Compute the ponder cost for ACT regularization.

    The ponder cost penalizes the model for using too many iterations.
    Following Graves (2016), it is the sum of (n_updates + remainders)
    across all tokens.

    This is added to the main loss with a small weight (e.g., 0.01) to
    encourage the model to halt early when possible.

    Args:
        act_state: Final ACT state after all iterations.

    Returns:
        Scalar ponder cost.
    """
    # N(t) + R(t) for each token, then mean across tokens
    per_token_cost = act_state.n_updates + act_state.remainders
    return jnp.mean(per_token_cost)


# ---------------------------------------------------------------------------
# Full ACT loop via jax.lax.scan (fixed max iterations with masking)
# ---------------------------------------------------------------------------

def adaptive_computation_time(
    transformer_block_fn,
    halting_mechanism: HaltingMechanism,
    initial_hidden: Float[Array, "seq_len d_model"],
    max_iterations: int,
    timestep_offset: int = 0,
) -> tuple[Float[Array, "seq_len d_model"], ACTState]:
    """Run adaptive computation time with a fixed max number of iterations.

    Uses jax.lax.scan for efficient compilation. At each iteration:
    1. Apply the transformer block to get updated hidden states
    2. Compute halting probabilities
    3. Accumulate weighted outputs
    4. Mask out halted tokens (they still go through the block but
       their contributions are zeroed)

    The transformer_block_fn should have signature:
        (hidden_states, timestep) -> updated_hidden_states
    where timestep is a scalar integer.

    Args:
        transformer_block_fn: Callable that applies one transformer block
            iteration. Signature: (hidden_states, timestep) -> hidden_states.
        halting_mechanism: HaltingMechanism module.
        initial_hidden: Initial hidden states, shape (seq_len, d_model).
        max_iterations: Maximum number of loop iterations.
        timestep_offset: Offset added to timestep indices (useful when
            chaining multiple ACT phases).

    Returns:
        Tuple of:
        - Final output hidden states (weighted combination), shape (seq_len, d_model).
        - Final ACT state (for computing ponder cost, inspecting n_updates, etc.).
    """
    seq_len, d_model = initial_hidden.shape

    # Initialize ACT state
    init_act_state = ACTState(
        halting_prob=jnp.zeros(seq_len),
        remainders=jnp.zeros(seq_len),
        n_updates=jnp.zeros(seq_len),
        halted=jnp.zeros(seq_len, dtype=jnp.bool_),
        accumulated_output=jnp.zeros_like(initial_hidden),
    )

    def scan_fn(carry, timestep):
        hidden, act_state = carry

        # Apply transformer block
        updated_hidden = transformer_block_fn(hidden, timestep + timestep_offset)

        # Mask: only update tokens that haven't halted
        still_running = ~act_state.halted
        # Blend: halted tokens keep their old hidden state
        masked_hidden = jnp.where(
            still_running[:, None],
            updated_hidden,
            hidden,
        )

        # ACT step
        new_act_state = act_step(halting_mechanism, masked_hidden, act_state)

        return (masked_hidden, new_act_state), new_act_state.n_updates

    timesteps = jnp.arange(max_iterations)
    (final_hidden, final_act_state), _ = jax.lax.scan(
        scan_fn,
        (initial_hidden, init_act_state),
        timesteps,
    )

    # For tokens that never reached the threshold within max_iterations,
    # assign the remainder to the last iteration
    still_running = ~final_act_state.halted
    remainder = 1.0 - final_act_state.halting_prob
    final_output = (
        final_act_state.accumulated_output
        + still_running.astype(jnp.float32)[:, None]
        * remainder[:, None]
        * final_hidden
    )

    # Update act_state to reflect forced halting
    final_act_state = ACTState(
        halting_prob=jnp.ones(seq_len),
        remainders=jnp.where(still_running, remainder, final_act_state.remainders),
        n_updates=final_act_state.n_updates,
        halted=jnp.ones(seq_len, dtype=jnp.bool_),
        accumulated_output=final_output,
    )

    return final_output, final_act_state


# ---------------------------------------------------------------------------
# Fixed loop count (no ACT, for comparison / simpler training)
# ---------------------------------------------------------------------------

def fixed_loop(
    transformer_block_fn,
    initial_hidden: Float[Array, "seq_len d_model"],
    n_iterations: int,
    timestep_offset: int = 0,
) -> Float[Array, "seq_len d_model"]:
    """Run the transformer block a fixed number of times via jax.lax.scan.

    No halting mechanism -- every token is updated for exactly n_iterations.
    This is simpler, faster, and recommended for initial experiments.

    Args:
        transformer_block_fn: Callable (hidden_states, timestep) -> hidden_states.
        initial_hidden: Initial hidden states, shape (seq_len, d_model).
        n_iterations: Number of loop iterations.
        timestep_offset: Offset for timestep indices.

    Returns:
        Final hidden states, shape (seq_len, d_model).
    """
    def scan_fn(hidden, timestep):
        updated = transformer_block_fn(hidden, timestep + timestep_offset)
        return updated, None

    final_hidden, _ = jax.lax.scan(
        scan_fn,
        initial_hidden,
        jnp.arange(n_iterations),
    )
    return final_hidden
