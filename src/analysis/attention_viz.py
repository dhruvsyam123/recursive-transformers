"""
Attention pattern visualization for Karatsuba looped transformer.

Provides tools to:
- Plot attention weights per head per loop iteration
- Highlight which input bits attend to which
- Save plots as PNG
- Compare attention patterns across loop iterations

This is crucial for mechanistic interpretability: understanding whether
different attention heads and loop iterations specialize for different
parts of the Karatsuba algorithm (splitting, base-case multiply,
addition, subtraction, combination).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

# Matplotlib with non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Extract attention weights from model
# ---------------------------------------------------------------------------

def extract_attention_weights(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    n_loops: int,
) -> List[jnp.ndarray]:
    """Extract attention weight matrices from each loop iteration.

    Runs the model forward and collects the attention pattern from each
    loop iteration. Requires the model to have an `extract_attention_weights`
    method or to expose attention weights through its forward pass.

    Args:
        model: LoopedTransformer with attention weight extraction support.
        tokens: (seq_len,) input tokens for a single example.
        positions: (seq_len, n_pos_dims) position ids.
        n_loops: Number of loop iterations.

    Returns:
        List of attention weight arrays, one per loop iteration.
        Each has shape (n_heads, seq_len, seq_len).
    """
    if hasattr(model, "forward_with_attention"):
        _, attention_weights = model.forward_with_attention(
            tokens, positions, n_loops
        )
        return attention_weights

    # Fallback: if the model doesn't expose attention weights directly,
    # we can try to intercept them by instrumenting the attention layer.
    # This is a common pattern in JAX/Equinox analysis code.
    print(
        "Warning: Model does not have 'forward_with_attention' method. "
        "Attention extraction requires the model to expose attention weights. "
        "Returning empty list."
    )
    return []


# ---------------------------------------------------------------------------
# Plot attention weights for a single head and iteration
# ---------------------------------------------------------------------------

def plot_attention_weights(
    attention_weights: jnp.ndarray,
    head_idx: int = 0,
    loop_idx: int = 0,
    token_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    show_values: bool = False,
    max_tokens: int = 64,
) -> plt.Figure:
    """Plot attention weights as a heatmap for a single head.

    Args:
        attention_weights: Either (n_heads, seq_len, seq_len) for a single
            iteration, or a list of such arrays indexed by loop_idx.
        head_idx: Which attention head to visualize.
        loop_idx: Which loop iteration (only used if attention_weights is a list).
        token_labels: Optional labels for each token position.
        title: Plot title. Auto-generated if None.
        save_path: Path to save PNG. If None, does not save.
        figsize: Figure size.
        cmap: Matplotlib colormap.
        show_values: If True, annotate cells with numeric values.
        max_tokens: Maximum number of tokens to display (truncate if longer).

    Returns:
        Matplotlib Figure object.
    """
    if isinstance(attention_weights, list):
        if loop_idx >= len(attention_weights):
            raise IndexError(
                f"loop_idx={loop_idx} but only {len(attention_weights)} iterations available"
            )
        attn = attention_weights[loop_idx]
    else:
        attn = attention_weights

    attn = np.array(attn)  # convert from JAX array

    if attn.ndim == 3:
        # (n_heads, seq_len, seq_len)
        if head_idx >= attn.shape[0]:
            raise IndexError(f"head_idx={head_idx} but only {attn.shape[0]} heads")
        attn_head = attn[head_idx]
    elif attn.ndim == 2:
        attn_head = attn
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    # Truncate if too long
    seq_len = attn_head.shape[0]
    if seq_len > max_tokens:
        attn_head = attn_head[:max_tokens, :max_tokens]
        if token_labels is not None:
            token_labels = token_labels[:max_tokens]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(attn_head, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    if title is None:
        title = f"Attention: Head {head_idx}, Loop {loop_idx}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Key position", fontsize=12)
    ax.set_ylabel("Query position", fontsize=12)

    if token_labels is not None and len(token_labels) <= 50:
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(token_labels)))
        ax.set_yticklabels(token_labels, fontsize=7)

    if show_values and attn_head.shape[0] <= 30:
        for i in range(attn_head.shape[0]):
            for j in range(attn_head.shape[1]):
                val = attn_head[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Plot attention across all loop iterations (comparison)
# ---------------------------------------------------------------------------

def plot_attention_across_iterations(
    attention_weights: List[jnp.ndarray],
    head_idx: int = 0,
    token_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize_per_panel: Tuple[float, float] = (5, 4),
    cmap: str = "Blues",
    max_tokens: int = 48,
    max_iterations: int = 8,
) -> plt.Figure:
    """Plot attention patterns side-by-side for each loop iteration.

    This reveals how attention patterns evolve across loop iterations,
    which should correspond to different recursion levels in the
    Karatsuba algorithm.

    Args:
        attention_weights: List of (n_heads, seq_len, seq_len) arrays.
        head_idx: Which attention head.
        token_labels: Token labels.
        title: Overall title.
        save_path: PNG save path.
        figsize_per_panel: Size of each sub-panel.
        cmap: Colormap.
        max_tokens: Max tokens per panel.
        max_iterations: Max iterations to show.

    Returns:
        Matplotlib Figure.
    """
    n_iters = min(len(attention_weights), max_iterations)
    if n_iters == 0:
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, "No attention weights available", ha="center", va="center")
        return fig

    cols = min(4, n_iters)
    rows = (n_iters + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )

    if title is None:
        title = f"Attention Evolution Across Loop Iterations (Head {head_idx})"
    fig.suptitle(title, fontsize=14, y=1.02)

    for i in range(n_iters):
        row, col = divmod(i, cols)
        ax = axes[row][col]

        attn = np.array(attention_weights[i])
        if attn.ndim == 3:
            attn_head = attn[head_idx]
        else:
            attn_head = attn

        seq_len = attn_head.shape[0]
        if seq_len > max_tokens:
            attn_head = attn_head[:max_tokens, :max_tokens]

        im = ax.imshow(attn_head, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Loop {i}", fontsize=10)

        if i % cols == 0:
            ax.set_ylabel("Query", fontsize=9)
        if row == rows - 1:
            ax.set_xlabel("Key", fontsize=9)

    # Hide unused axes
    for i in range(n_iters, rows * cols):
        row, col = divmod(i, cols)
        axes[row][col].set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Highlight which input bits attend to which
# ---------------------------------------------------------------------------

def highlight_bit_attention(
    attention_weights: jnp.ndarray,
    input_a_range: Tuple[int, int],
    input_b_range: Tuple[int, int],
    output_range: Tuple[int, int],
    head_idx: int = 0,
    loop_idx: int = 0,
    token_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Visualize attention between specific regions (input A, input B, output).

    Creates a focused view showing:
    - How output bits attend to input A bits
    - How output bits attend to input B bits
    - Cross-attention between the two inputs

    This is useful for understanding whether the model has learned to
    associate the correct input bits for each sub-problem.

    Args:
        attention_weights: Attention weights (list or single array).
        input_a_range: (start, end) token positions for operand A.
        input_b_range: (start, end) token positions for operand B.
        output_range: (start, end) token positions for the output.
        head_idx: Attention head index.
        loop_idx: Loop iteration index.
        token_labels: Optional token labels.
        title: Plot title.
        save_path: PNG save path.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(attention_weights, list):
        attn = np.array(attention_weights[loop_idx])
    else:
        attn = np.array(attention_weights)

    if attn.ndim == 3:
        attn = attn[head_idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    a_start, a_end = input_a_range
    b_start, b_end = input_b_range
    o_start, o_end = output_range

    # Panel 1: Output attending to Input A
    region_oa = attn[o_start:o_end, a_start:a_end]
    im1 = axes[0].imshow(region_oa, cmap="Reds", aspect="auto", vmin=0)
    axes[0].set_title("Output -> Input A", fontsize=11)
    axes[0].set_xlabel("Input A position")
    axes[0].set_ylabel("Output position")
    fig.colorbar(im1, ax=axes[0], fraction=0.046)

    # Panel 2: Output attending to Input B
    region_ob = attn[o_start:o_end, b_start:b_end]
    im2 = axes[1].imshow(region_ob, cmap="Blues", aspect="auto", vmin=0)
    axes[1].set_title("Output -> Input B", fontsize=11)
    axes[1].set_xlabel("Input B position")
    axes[1].set_ylabel("Output position")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    # Panel 3: Input A attending to Input B (cross-attention)
    region_ab = attn[a_start:a_end, b_start:b_end]
    im3 = axes[2].imshow(region_ab, cmap="Purples", aspect="auto", vmin=0)
    axes[2].set_title("Input A -> Input B", fontsize=11)
    axes[2].set_xlabel("Input B position")
    axes[2].set_ylabel("Input A position")
    fig.colorbar(im3, ax=axes[2], fraction=0.046)

    if title is None:
        title = f"Bit-level Attention: Head {head_idx}, Loop {loop_idx}"
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Attention head specialization analysis
# ---------------------------------------------------------------------------

def analyze_head_specialization(
    attention_weights: List[jnp.ndarray],
    input_a_range: Tuple[int, int],
    input_b_range: Tuple[int, int],
    output_range: Tuple[int, int],
) -> Dict[str, Any]:
    """Analyze whether attention heads specialize for different operations.

    Computes summary statistics for each head across iterations:
    - How much each head attends to inputs vs. intermediate results
    - Whether heads show "diagonal" patterns (local computation)
    - Whether heads show cross-region patterns (recombination)

    Args:
        attention_weights: List of (n_heads, seq_len, seq_len) arrays.
        input_a_range: Token range for operand A.
        input_b_range: Token range for operand B.
        output_range: Token range for output.

    Returns:
        Dict with per-head specialization metrics.
    """
    if not attention_weights:
        return {"error": "No attention weights provided"}

    n_iters = len(attention_weights)
    attn_0 = np.array(attention_weights[0])
    n_heads = attn_0.shape[0] if attn_0.ndim == 3 else 1

    a_start, a_end = input_a_range
    b_start, b_end = input_b_range
    o_start, o_end = output_range

    head_stats = {}

    for h in range(n_heads):
        head_stats[h] = {
            "input_a_attention_per_iter": [],
            "input_b_attention_per_iter": [],
            "output_self_attention_per_iter": [],
            "diagonal_strength_per_iter": [],
        }

        for it in range(n_iters):
            attn = np.array(attention_weights[it])
            if attn.ndim == 3:
                attn_h = attn[h]
            else:
                attn_h = attn

            # How much the output attends to input A
            out_to_a = np.mean(attn_h[o_start:o_end, a_start:a_end])
            head_stats[h]["input_a_attention_per_iter"].append(float(out_to_a))

            # How much the output attends to input B
            out_to_b = np.mean(attn_h[o_start:o_end, b_start:b_end])
            head_stats[h]["input_b_attention_per_iter"].append(float(out_to_b))

            # Self-attention within output region
            out_self = np.mean(attn_h[o_start:o_end, o_start:o_end])
            head_stats[h]["output_self_attention_per_iter"].append(float(out_self))

            # Diagonal strength (measure of locality)
            diag_mask = np.eye(attn_h.shape[0], attn_h.shape[1])
            diag_strength = np.mean(attn_h * diag_mask) / max(np.mean(attn_h), 1e-8)
            head_stats[h]["diagonal_strength_per_iter"].append(float(diag_strength))

    return head_stats


# ---------------------------------------------------------------------------
# Attention entropy analysis
# ---------------------------------------------------------------------------

def attention_entropy(
    attention_weights: List[jnp.ndarray],
) -> Dict[str, Any]:
    """Compute entropy of attention distributions per head per iteration.

    Low entropy = concentrated attention (specialized).
    High entropy = diffuse attention (broadcast).

    Args:
        attention_weights: List of (n_heads, seq_len, seq_len) arrays.

    Returns:
        Dict with entropy statistics.
    """
    results = {"per_iteration": []}

    for it, attn in enumerate(attention_weights):
        attn = np.array(attn)
        if attn.ndim == 3:
            n_heads = attn.shape[0]
        else:
            n_heads = 1
            attn = attn[np.newaxis]

        iter_results = {}
        for h in range(n_heads):
            # Entropy of each row (query position's attention distribution)
            attn_h = attn[h]
            # Clamp to avoid log(0)
            attn_h = np.clip(attn_h, 1e-10, 1.0)
            entropy = -np.sum(attn_h * np.log2(attn_h), axis=-1)
            iter_results[f"head_{h}_mean_entropy"] = float(np.mean(entropy))
            iter_results[f"head_{h}_max_entropy"] = float(np.max(entropy))
            iter_results[f"head_{h}_min_entropy"] = float(np.min(entropy))

        results["per_iteration"].append(iter_results)

    return results


# ---------------------------------------------------------------------------
# Batch visualization utility
# ---------------------------------------------------------------------------

def save_all_attention_plots(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    n_loops: int,
    output_dir: str,
    token_labels: Optional[List[str]] = None,
    max_heads: int = 8,
):
    """Generate and save attention plots for all heads and iterations.

    Creates a directory structure:
        output_dir/
            head_0/
                loop_0.png
                loop_1.png
                ...
            head_1/
                ...
            comparison/
                head_0_all_loops.png
                ...

    Args:
        model: Trained model.
        tokens: (seq_len,) single example tokens.
        positions: (seq_len, n_pos_dims) positions.
        n_loops: Loop iterations.
        output_dir: Base output directory.
        token_labels: Optional token labels.
        max_heads: Maximum number of heads to plot.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract attention weights
    attn_weights = extract_attention_weights(model, tokens, positions, n_loops)
    if not attn_weights:
        print("Could not extract attention weights.")
        return

    n_iters = len(attn_weights)
    attn_0 = np.array(attn_weights[0])
    n_heads = attn_0.shape[0] if attn_0.ndim == 3 else 1
    n_heads = min(n_heads, max_heads)

    # Individual plots
    for h in range(n_heads):
        head_dir = output_path / f"head_{h}"
        head_dir.mkdir(exist_ok=True)
        for it in range(n_iters):
            fig = plot_attention_weights(
                attn_weights,
                head_idx=h,
                loop_idx=it,
                token_labels=token_labels,
                title=f"Head {h}, Loop Iteration {it}",
                save_path=str(head_dir / f"loop_{it}.png"),
            )
            plt.close(fig)

    # Comparison plots (all iterations for each head)
    comp_dir = output_path / "comparison"
    comp_dir.mkdir(exist_ok=True)
    for h in range(n_heads):
        fig = plot_attention_across_iterations(
            attn_weights,
            head_idx=h,
            token_labels=token_labels,
            title=f"Head {h}: Attention Across All Loop Iterations",
            save_path=str(comp_dir / f"head_{h}_all_loops.png"),
        )
        plt.close(fig)

    print(f"Attention plots saved to: {output_dir}")
    print(f"  {n_heads} heads x {n_iters} iterations = {n_heads * n_iters} individual plots")
    print(f"  {n_heads} comparison plots")
