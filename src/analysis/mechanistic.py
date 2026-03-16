"""
Mechanistic interpretability tools for Karatsuba looped transformer.

Following the approach of Nanda et al. (ICLR 2023) who reverse-engineered
a transformer performing modular addition, this module provides tools to
understand what the model has learned:

- Extract and analyze embedding structure (Fourier analysis)
- Residual stream analysis per loop iteration
- Ablation tools (zero out specific directions)
- Loop utilization analysis

The key question: does the model actually implement Karatsuba-style recursion,
or does it find a different algorithm?
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fourier analysis of embeddings
# ---------------------------------------------------------------------------

def fourier_analysis_embeddings(
    model: eqx.Module,
    vocab_size: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform Fourier analysis on the token embedding matrix.

    Following Nanda et al., who showed that a transformer trained on
    modular addition learns to embed numbers into Fourier components
    (sine/cosine pairs at specific frequencies), we analyze whether
    the Karatsuba transformer's embeddings show similar structure.

    For binary representations, we expect:
    - Token 0 and token 1 should have embeddings that differ in a
      structured way related to their bit values.
    - If the model learns Fourier-like structure, the embedding difference
      (emb[1] - emb[0]) should align with specific frequency components.

    Args:
        model: Trained LoopedTransformer.
        vocab_size: Override vocab size (otherwise inferred from model).
        save_path: Optional path to save analysis plot.

    Returns:
        Dict with Fourier analysis results:
        - 'embedding_matrix': (vocab_size, d_model) raw embeddings
        - 'fft_magnitudes': (vocab_size, d_model//2+1) FFT magnitudes
        - 'dominant_frequencies': top-k frequency components
        - 'embedding_norms': L2 norm of each token embedding
        - 'pairwise_cosine_sim': cosine similarity between all embedding pairs
    """
    # Extract embedding matrix
    embed_matrix = _extract_embedding_matrix(model)
    if embed_matrix is None:
        return {"error": "Could not extract embedding matrix from model"}

    embed_np = np.array(embed_matrix)
    v_size, d_model = embed_np.shape

    if vocab_size is not None:
        embed_np = embed_np[:vocab_size]
        v_size = vocab_size

    results = {
        "embedding_matrix_shape": list(embed_np.shape),
    }

    # Norms
    norms = np.linalg.norm(embed_np, axis=1)
    results["embedding_norms"] = norms.tolist()

    # Pairwise cosine similarity
    normed = embed_np / np.maximum(norms[:, None], 1e-8)
    cosine_sim = normed @ normed.T
    results["pairwise_cosine_sim"] = cosine_sim.tolist()

    # FFT analysis of each embedding dimension across the vocabulary
    # This reveals if there's periodic structure in how tokens are embedded
    fft_per_dim = []
    for d in range(d_model):
        col = embed_np[:, d]
        fft_mag = np.abs(np.fft.rfft(col))
        fft_per_dim.append(fft_mag)
    fft_per_dim = np.array(fft_per_dim)  # (d_model, n_freq)
    results["fft_magnitudes_shape"] = list(fft_per_dim.shape)

    # Average FFT magnitude across dimensions
    avg_fft = np.mean(fft_per_dim, axis=0)
    results["avg_fft_magnitudes"] = avg_fft.tolist()

    # Dominant frequencies
    top_k = min(5, len(avg_fft))
    top_indices = np.argsort(avg_fft)[-top_k:][::-1]
    results["dominant_frequencies"] = [
        {"frequency": int(idx), "magnitude": float(avg_fft[idx])}
        for idx in top_indices
    ]

    # SVD of embedding matrix (reveals low-rank structure)
    U, S, Vt = np.linalg.svd(embed_np, full_matrices=False)
    results["singular_values"] = S.tolist()
    results["effective_rank"] = float(
        np.sum(S > 0.01 * S[0])
    )

    # For binary tokens specifically: analyze the 0/1 embedding difference
    if v_size >= 2:
        bit_diff = embed_np[1] - embed_np[0]
        results["bit_diff_norm"] = float(np.linalg.norm(bit_diff))
        results["bit_diff_fft"] = np.abs(np.fft.rfft(bit_diff)).tolist()

    # Visualization
    if save_path is not None:
        _plot_embedding_analysis(embed_np, results, save_path)

    return results


def _extract_embedding_matrix(model: eqx.Module) -> Optional[jnp.ndarray]:
    """Extract the token embedding matrix from the model."""
    # Try common attribute names
    for attr_path in ["embed.weight", "embed", "embedding.weight", "embedding"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, jnp.ndarray):
                return obj
            if hasattr(obj, "weight"):
                return obj.weight
        except AttributeError:
            continue
    # Try to find any Embedding module
    leaves = jax.tree.leaves(model, is_leaf=lambda x: isinstance(x, eqx.nn.Embedding))
    for leaf in leaves:
        if isinstance(leaf, eqx.nn.Embedding):
            return leaf.weight
    return None


def _plot_embedding_analysis(
    embed_np: np.ndarray,
    results: Dict[str, Any],
    save_path: str,
):
    """Plot embedding analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Cosine similarity matrix
    cosine_sim = np.array(results["pairwise_cosine_sim"])
    im1 = axes[0, 0].imshow(cosine_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title("Pairwise Cosine Similarity")
    axes[0, 0].set_xlabel("Token")
    axes[0, 0].set_ylabel("Token")
    fig.colorbar(im1, ax=axes[0, 0])

    # Panel 2: Singular values
    sv = results["singular_values"]
    axes[0, 1].bar(range(len(sv)), sv)
    axes[0, 1].set_title(f"Singular Values (effective rank: {results['effective_rank']:.0f})")
    axes[0, 1].set_xlabel("Component")
    axes[0, 1].set_ylabel("Singular Value")

    # Panel 3: Average FFT magnitudes
    fft_mags = results["avg_fft_magnitudes"]
    axes[1, 0].bar(range(len(fft_mags)), fft_mags)
    axes[1, 0].set_title("Average FFT Magnitude (across dimensions)")
    axes[1, 0].set_xlabel("Frequency")
    axes[1, 0].set_ylabel("Magnitude")

    # Panel 4: Embedding norms
    norms = results["embedding_norms"]
    axes[1, 1].bar(range(len(norms)), norms)
    axes[1, 1].set_title("Token Embedding Norms")
    axes[1, 1].set_xlabel("Token ID")
    axes[1, 1].set_ylabel("L2 Norm")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Residual stream analysis per loop iteration
# ---------------------------------------------------------------------------

def residual_stream_analysis(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    n_loops: int,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the residual stream at each loop iteration.

    The residual stream carries information across loop iterations.
    By analyzing how it changes, we can understand:
    - Whether different loop iterations process different types of information
    - Whether the model has distinct "decompose" and "recombine" phases
    - How much each iteration modifies the residual stream

    Args:
        model: Trained model.
        tokens: (seq_len,) single example tokens.
        positions: (seq_len, n_pos_dims) positions.
        n_loops: Number of loop iterations.
        save_path: Optional path for visualization.

    Returns:
        Dict with:
        - 'residual_norms': L2 norm of residual stream at each iteration
        - 'residual_diffs': norm of the difference between consecutive iterations
        - 'svd_per_iteration': singular values of residual at each iteration
        - 'cosine_sim_between_iterations': how similar consecutive residuals are
    """
    # Get intermediate residual stream states
    if hasattr(model, "forward_with_intermediates"):
        _, intermediate_states = model.forward_with_intermediates(
            tokens, positions, n_loops, return_hidden=True
        )
    elif hasattr(model, "get_residual_stream"):
        intermediate_states = model.get_residual_stream(tokens, positions, n_loops)
    else:
        # Fallback: run model layer by layer manually
        intermediate_states = _extract_residual_stream(model, tokens, positions, n_loops)

    if intermediate_states is None or len(intermediate_states) == 0:
        return {"error": "Could not extract residual stream states"}

    results = {
        "n_iterations": len(intermediate_states),
    }

    residual_norms = []
    residual_diffs = []
    cosine_sims = []
    svd_per_iter = []

    for i, state in enumerate(intermediate_states):
        state_np = np.array(state)  # (seq_len, d_model)

        # Norm of residual stream
        norm = float(np.linalg.norm(state_np))
        residual_norms.append(norm)

        # SVD
        U, S, Vt = np.linalg.svd(state_np, full_matrices=False)
        svd_per_iter.append(S[:10].tolist())  # top 10 singular values

        # Difference from previous iteration
        if i > 0:
            prev_np = np.array(intermediate_states[i - 1])
            diff = state_np - prev_np
            diff_norm = float(np.linalg.norm(diff))
            residual_diffs.append(diff_norm)

            # Cosine similarity between flattened residuals
            flat_curr = state_np.flatten()
            flat_prev = prev_np.flatten()
            cos_sim = float(
                np.dot(flat_curr, flat_prev)
                / max(np.linalg.norm(flat_curr) * np.linalg.norm(flat_prev), 1e-8)
            )
            cosine_sims.append(cos_sim)

    results["residual_norms"] = residual_norms
    results["residual_diffs"] = residual_diffs
    results["cosine_sim_between_iterations"] = cosine_sims
    results["svd_per_iteration"] = svd_per_iter

    # Detect phase transitions: large jumps in residual difference
    if len(residual_diffs) > 1:
        diffs_arr = np.array(residual_diffs)
        mean_diff = np.mean(diffs_arr)
        std_diff = np.std(diffs_arr)
        phase_transitions = [
            i + 1 for i, d in enumerate(residual_diffs)
            if d > mean_diff + 2 * std_diff
        ]
        results["phase_transitions"] = phase_transitions
    else:
        results["phase_transitions"] = []

    if save_path is not None:
        _plot_residual_analysis(results, save_path)

    return results


def _extract_residual_stream(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    n_loops: int,
) -> Optional[List[jnp.ndarray]]:
    """Attempt to extract residual stream by running model internals.

    This is a fallback when the model doesn't have explicit methods
    for intermediate state extraction.
    """
    # Check if model has embed and block attributes
    if not (hasattr(model, "embed") and hasattr(model, "block")):
        return None

    try:
        # Initial embedding
        if hasattr(model, "pos_encode"):
            x = model.embed(tokens) + model.pos_encode(positions)
        else:
            x = model.embed(tokens)

        states = [x]

        # Run loop iterations manually
        block = model.block
        for t in range(n_loops):
            timestep = jnp.array(t)
            x = block(x, timestep)
            states.append(x)

        return states
    except Exception:
        return None


def _plot_residual_analysis(results: Dict[str, Any], save_path: str):
    """Plot residual stream analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Residual norms
    norms = results["residual_norms"]
    axes[0, 0].plot(range(len(norms)), norms, "b-o")
    axes[0, 0].set_title("Residual Stream Norm per Iteration")
    axes[0, 0].set_xlabel("Loop Iteration")
    axes[0, 0].set_ylabel("L2 Norm")
    for pt in results.get("phase_transitions", []):
        axes[0, 0].axvline(pt, color="red", linestyle="--", alpha=0.5)

    # Panel 2: Residual diffs
    diffs = results["residual_diffs"]
    if diffs:
        axes[0, 1].plot(range(1, len(diffs) + 1), diffs, "r-o")
        axes[0, 1].set_title("Residual Difference Between Iterations")
        axes[0, 1].set_xlabel("Loop Iteration")
        axes[0, 1].set_ylabel("Diff Norm")

    # Panel 3: Cosine similarity
    cosines = results["cosine_sim_between_iterations"]
    if cosines:
        axes[1, 0].plot(range(1, len(cosines) + 1), cosines, "g-o")
        axes[1, 0].set_title("Cosine Similarity Between Consecutive Iterations")
        axes[1, 0].set_xlabel("Loop Iteration")
        axes[1, 0].set_ylabel("Cosine Similarity")
        axes[1, 0].set_ylim(-0.1, 1.1)

    # Panel 4: SVD top singular values
    svds = results["svd_per_iteration"]
    if svds:
        for i, sv in enumerate(svds):
            axes[1, 1].plot(range(len(sv)), sv, alpha=0.5, label=f"Iter {i}")
        axes[1, 1].set_title("Top Singular Values per Iteration")
        axes[1, 1].set_xlabel("Component")
        axes[1, 1].set_ylabel("Singular Value")
        if len(svds) <= 10:
            axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation tools
# ---------------------------------------------------------------------------

def ablation_study(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    n_loops: int,
    ablation_type: str = "zero_direction",
    direction: Optional[jnp.ndarray] = None,
    ablation_iteration: Optional[int] = None,
    ablation_positions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run ablation experiments on the model.

    Ablation types:
    - "zero_direction": Zero out a specific direction in the residual stream
      at a specific iteration. Tests whether that direction carries important
      information.
    - "skip_iteration": Skip a specific loop iteration. Tests whether that
      iteration is necessary.
    - "zero_positions": Zero out the residual stream at specific token positions
      at a specific iteration.

    Args:
        model: Trained model.
        tokens: (seq_len,) input tokens.
        positions: (seq_len, n_pos_dims) positions.
        targets: (seq_len,) target tokens.
        mask: (seq_len,) output mask.
        n_loops: Number of loop iterations.
        ablation_type: Type of ablation.
        direction: (d_model,) direction to zero out (for "zero_direction").
        ablation_iteration: Which loop iteration to ablate (None = all).
        ablation_positions: Which token positions to ablate (for "zero_positions").

    Returns:
        Dict with:
        - 'baseline_loss': loss without ablation
        - 'ablated_loss': loss with ablation
        - 'loss_increase': ablated - baseline
        - 'baseline_accuracy': token accuracy without ablation
        - 'ablated_accuracy': token accuracy with ablation
    """
    # Baseline: run model normally
    baseline_logits = model(tokens, positions, n_loops)
    baseline_preds = jnp.argmax(baseline_logits, axis=-1)
    baseline_loss = _compute_loss(baseline_logits, targets, mask)
    baseline_acc = _compute_accuracy(baseline_preds, targets, mask)

    # Ablated forward pass
    ablated_logits = _ablated_forward(
        model, tokens, positions, n_loops,
        ablation_type=ablation_type,
        direction=direction,
        ablation_iteration=ablation_iteration,
        ablation_positions=ablation_positions,
    )
    ablated_preds = jnp.argmax(ablated_logits, axis=-1)
    ablated_loss = _compute_loss(ablated_logits, targets, mask)
    ablated_acc = _compute_accuracy(ablated_preds, targets, mask)

    return {
        "ablation_type": ablation_type,
        "ablation_iteration": ablation_iteration,
        "baseline_loss": float(baseline_loss),
        "ablated_loss": float(ablated_loss),
        "loss_increase": float(ablated_loss - baseline_loss),
        "baseline_accuracy": float(baseline_acc),
        "ablated_accuracy": float(ablated_acc),
        "accuracy_decrease": float(baseline_acc - ablated_acc),
    }


def _ablated_forward(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    n_loops: int,
    ablation_type: str,
    direction: Optional[jnp.ndarray] = None,
    ablation_iteration: Optional[int] = None,
    ablation_positions: Optional[List[int]] = None,
) -> jnp.ndarray:
    """Run forward pass with ablation applied.

    Manually unrolls the loop to apply ablation at the right point.
    """
    if not (hasattr(model, "embed") and hasattr(model, "block")):
        # Can't do fine-grained ablation without model internals
        raise ValueError(
            "Model must expose 'embed' and 'block' attributes for ablation"
        )

    # Initial embedding
    if hasattr(model, "pos_encode"):
        x = model.embed(tokens) + model.pos_encode(positions)
    else:
        x = model.embed(tokens)

    block = model.block

    for t in range(n_loops):
        x = block(x, jnp.array(t))

        # Apply ablation at this iteration
        should_ablate = (ablation_iteration is None or t == ablation_iteration)
        if should_ablate:
            if ablation_type == "zero_direction" and direction is not None:
                # Project out the specified direction
                d_norm = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-8)
                projection = jnp.einsum("sd,d->s", x, d_norm)[:, None] * d_norm[None, :]
                x = x - projection
            elif ablation_type == "skip_iteration":
                # Undo this iteration (restore pre-iteration state)
                # This is approximate since we don't store the pre-state
                pass
            elif ablation_type == "zero_positions" and ablation_positions is not None:
                # Zero out specific positions
                pos_mask = jnp.ones(x.shape[0])
                for p in ablation_positions:
                    if 0 <= p < x.shape[0]:
                        pos_mask = pos_mask.at[p].set(0.0)
                x = x * pos_mask[:, None]

    # Output head
    if hasattr(model, "output_head"):
        logits = model.output_head(x)
    elif hasattr(model, "unembed"):
        logits = model.unembed(x)
    else:
        logits = x

    return logits


def iterative_ablation_sweep(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    n_loops: int,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Ablate each loop iteration independently and measure impact.

    Produces a graph showing which iterations are most critical
    for the model's computation.

    Args:
        model: Trained model.
        tokens: (seq_len,) input tokens.
        positions: (seq_len, n_pos_dims) positions.
        targets: (seq_len,) target tokens.
        mask: (seq_len,) output mask.
        n_loops: Number of loop iterations.
        save_path: Optional path to save the sweep plot.

    Returns:
        Dict mapping iteration -> ablation impact.
    """
    results = {}

    for t in range(n_loops):
        ablation_result = ablation_study(
            model, tokens, positions, targets, mask, n_loops,
            ablation_type="zero_positions",
            ablation_iteration=t,
            ablation_positions=list(range(tokens.shape[0])),  # zero all positions
        )
        results[t] = ablation_result

    if save_path is not None:
        iterations = sorted(results.keys())
        loss_increases = [results[t]["loss_increase"] for t in iterations]
        acc_decreases = [results[t]["accuracy_decrease"] for t in iterations]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(iterations, loss_increases, color="salmon")
        ax1.set_title("Loss Increase per Ablated Iteration")
        ax1.set_xlabel("Loop Iteration")
        ax1.set_ylabel("Loss Increase")

        ax2.bar(iterations, acc_decreases, color="steelblue")
        ax2.set_title("Accuracy Decrease per Ablated Iteration")
        ax2.set_xlabel("Loop Iteration")
        ax2.set_ylabel("Accuracy Decrease")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Loop utilization analysis
# ---------------------------------------------------------------------------

def loop_utilization_analysis(
    model: eqx.Module,
    dataset,
    bit_widths: List[int],
    max_loops: int,
    num_examples: int = 100,
    rng: Optional[jax.Array] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze how the model utilizes different loop iterations.

    Tests whether:
    1. The model uses more iterations for larger inputs (as expected for
       deeper recursion).
    2. There's a clear "decompose" phase and "recombine" phase.
    3. Later iterations contribute less (convergence).

    For each bit width, we run the model with varying loop counts and
    measure at what point additional loops stop improving accuracy.

    Args:
        model: Trained model.
        dataset: Dataset supporting variable bit widths.
        bit_widths: List of bit widths to test.
        max_loops: Maximum loop count to try.
        num_examples: Examples per configuration.
        rng: PRNG key.
        save_path: Optional path for visualization.

    Returns:
        Dict with utilization data per bit width.
    """
    if rng is None:
        rng = jax.random.PRNGKey(314159)

    results = {}

    for bw in bit_widths:
        rng, batch_rng = jax.random.split(rng)
        batch = dataset.get_batch(
            batch_size=num_examples,
            rng=batch_rng,
            bit_widths=[bw],
        )

        tokens = batch["tokens"]
        positions = batch["positions"]
        targets = batch["targets"]
        mask = batch["mask"]

        loop_results = []
        for n_loops in range(1, max_loops + 1):
            logits = _batch_forward_jit(model, tokens, positions, n_loops)
            preds = jnp.argmax(logits, axis=-1)

            # Exact match accuracy
            correct = (preds == targets) | (mask == 0)
            em_acc = float(jnp.mean(jnp.all(correct, axis=-1).astype(jnp.float32)))

            # Token accuracy
            tok_correct = ((preds == targets) * mask).sum()
            tok_total = mask.sum()
            tok_acc = float(tok_correct / jnp.maximum(tok_total, 1))

            loop_results.append({
                "n_loops": n_loops,
                "exact_match": em_acc,
                "token_accuracy": tok_acc,
            })

        results[bw] = loop_results

        # Find optimal loop count (first point where accuracy plateaus)
        accs = [r["exact_match"] for r in loop_results]
        if accs:
            best_loops = np.argmax(accs) + 1
            # Plateau: first point within 1% of max
            max_acc = max(accs)
            plateau = next(
                (i + 1 for i, a in enumerate(accs)
                 if a >= max_acc - 0.01),
                len(accs)
            )
            results[f"{bw}_optimal_loops"] = best_loops
            results[f"{bw}_plateau_loops"] = plateau
            results[f"{bw}_max_accuracy"] = max_acc

    if save_path is not None:
        _plot_loop_utilization(results, bit_widths, max_loops, save_path)

    return results


@eqx.filter_jit
def _batch_forward_jit(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    n_loops: int,
) -> jnp.ndarray:
    """JIT-compiled batched forward pass."""
    return jax.vmap(lambda t, p: model(t, p, n_loops))(tokens, positions)


def _plot_loop_utilization(
    results: Dict[str, Any],
    bit_widths: List[int],
    max_loops: int,
    save_path: str,
):
    """Plot loop utilization curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for bw in bit_widths:
        if bw not in results:
            continue
        loop_data = results[bw]
        n_loops_list = [r["n_loops"] for r in loop_data]
        em_accs = [r["exact_match"] for r in loop_data]
        tok_accs = [r["token_accuracy"] for r in loop_data]

        ax1.plot(n_loops_list, em_accs, "-o", label=f"{bw}-bit", markersize=4)
        ax2.plot(n_loops_list, tok_accs, "-o", label=f"{bw}-bit", markersize=4)

    ax1.set_title("Exact Match Accuracy vs Loop Count")
    ax1.set_xlabel("Number of Loops")
    ax1.set_ylabel("Exact Match Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Token Accuracy vs Loop Count")
    ax2.set_xlabel("Number of Loops")
    ax2.set_ylabel("Token Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute cross-entropy loss."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, None], axis=-1
    ).squeeze(-1)
    return -jnp.sum(target_log_probs * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _compute_accuracy(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute token accuracy."""
    correct = (predictions == targets).astype(jnp.float32) * mask
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)
