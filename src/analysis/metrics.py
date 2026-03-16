"""
Metrics computation for Karatsuba looped transformer.

Provides:
- Token-level accuracy
- Sequence-level exact match
- Per-bit-position accuracy
- Carry propagation error analysis
- Loss per recursion level
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Token-level accuracy
# ---------------------------------------------------------------------------

def token_accuracy(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> float:
    """Compute token-level accuracy (fraction of correctly predicted tokens).

    Args:
        predictions: (...) integer predictions.
        targets: (...) integer targets (same shape as predictions).
        mask: Optional (...) binary mask. If provided, only masked positions
            contribute to accuracy.

    Returns:
        Scalar accuracy in [0, 1].
    """
    correct = (predictions == targets).astype(jnp.float32)
    if mask is not None:
        correct = correct * mask
        total = jnp.sum(mask)
    else:
        total = jnp.float32(correct.size)
    return float(jnp.sum(correct) / jnp.maximum(total, 1.0))


# ---------------------------------------------------------------------------
# Sequence-level exact match
# ---------------------------------------------------------------------------

def sequence_exact_match(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> float:
    """Compute sequence-level exact match accuracy.

    A sequence is considered correct only if ALL masked positions match.

    Args:
        predictions: (batch, seq_len) integer predictions.
        targets: (batch, seq_len) integer targets.
        mask: (batch, seq_len) binary mask.

    Returns:
        Fraction of sequences that are entirely correct.
    """
    if mask is not None:
        # Positions outside the mask are always "correct"
        correct_or_masked = (predictions == targets) | (mask == 0)
    else:
        correct_or_masked = predictions == targets

    all_correct = jnp.all(correct_or_masked, axis=-1)  # (batch,)
    return float(jnp.mean(all_correct.astype(jnp.float32)))


# ---------------------------------------------------------------------------
# Per-bit-position accuracy
# ---------------------------------------------------------------------------

def per_bit_position_accuracy(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    bit_significance: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Compute accuracy at each bit position in the output.

    This reveals where errors concentrate. Typical patterns:
    - LSB errors: base case mistakes
    - MSB errors: carry propagation failures
    - Errors at recursion boundaries: failure to combine sub-results

    Args:
        predictions: (batch, seq_len) predicted tokens.
        targets: (batch, seq_len) target tokens.
        mask: (batch, seq_len) output mask.
        bit_significance: (batch, seq_len) optional bit position indices.
            If provided, accuracy is reported per bit significance rather
            than per sequence position.

    Returns:
        Dict with:
        - 'per_position': list of (position, accuracy) tuples
        - 'lsb_accuracy': accuracy of least significant quarter
        - 'msb_accuracy': accuracy of most significant quarter
        - 'mid_accuracy': accuracy of middle half
    """
    correct = (predictions == targets).astype(jnp.float32) * mask

    if bit_significance is not None:
        # Group by bit significance
        max_bit = int(jnp.max(bit_significance)) + 1
        per_bit = {}
        for b in range(max_bit):
            bit_mask = (bit_significance == b).astype(jnp.float32) * mask
            bit_correct = jnp.sum(correct * (bit_significance == b).astype(jnp.float32))
            bit_total = jnp.sum(bit_mask)
            if float(bit_total) > 0:
                per_bit[b] = float(bit_correct / bit_total)
            else:
                per_bit[b] = None
        position_accs = [(b, acc) for b, acc in per_bit.items() if acc is not None]
    else:
        # Per sequence position
        pos_correct = jnp.sum(correct, axis=0)  # (seq_len,)
        pos_total = jnp.sum(mask, axis=0)
        pos_acc = jnp.where(pos_total > 0, pos_correct / pos_total, 0.0)
        position_accs = [(i, float(pos_acc[i])) for i in range(len(pos_acc))
                         if float(pos_total[i]) > 0]

    # Divide into LSB / mid / MSB regions
    n_positions = len(position_accs)
    quarter = max(1, n_positions // 4)

    if n_positions > 0:
        accs = [a for _, a in position_accs]
        lsb_acc = sum(accs[:quarter]) / quarter
        msb_acc = sum(accs[-quarter:]) / quarter
        mid_acc = sum(accs[quarter:-quarter]) / max(len(accs[quarter:-quarter]), 1)
    else:
        lsb_acc = msb_acc = mid_acc = 0.0

    return {
        "per_position": position_accs,
        "lsb_accuracy": lsb_acc,
        "msb_accuracy": msb_acc,
        "mid_accuracy": mid_acc,
        "num_positions": n_positions,
    }


# ---------------------------------------------------------------------------
# Carry propagation error analysis
# ---------------------------------------------------------------------------

def carry_propagation_errors(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    input_a: Optional[jnp.ndarray] = None,
    input_b: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Analyze carry propagation errors in multiplication output.

    Carry propagation is one of the hardest aspects of multiplication for
    transformers. This function identifies cases where:
    1. An error at position i is preceded by a correct position i-1
       (carry error originating at position i)
    2. Consecutive error runs (carry propagation chains)
    3. Correlation between input bit patterns and carry errors

    Args:
        predictions: (batch, seq_len) predicted output bits.
        targets: (batch, seq_len) target output bits.
        mask: (batch, seq_len) output mask.
        input_a: (batch, input_len) first operand bits (optional).
        input_b: (batch, input_len) second operand bits (optional).

    Returns:
        Dict with carry error statistics.
    """
    errors = ((predictions != targets) & (mask == 1)).astype(jnp.float32)

    # Error run analysis: consecutive errors indicate carry propagation failure
    batch_size, seq_len = errors.shape

    # Shifted errors for detecting run boundaries
    errors_shifted = jnp.concatenate(
        [jnp.zeros((batch_size, 1)), errors[:, :-1]], axis=1
    )

    # "New" errors: position is wrong but previous position was correct
    # These are likely the origination points of carry errors
    new_errors = errors * (1.0 - errors_shifted) * mask
    # "Propagated" errors: both this and previous position are wrong
    propagated = errors * errors_shifted * mask

    n_new_errors = float(jnp.sum(new_errors))
    n_propagated = float(jnp.sum(propagated))
    n_total_errors = float(jnp.sum(errors))

    # Run length distribution
    # Count consecutive error runs per example
    run_lengths = []
    for b in range(min(batch_size, 100)):  # sample up to 100 examples
        example_errors = errors[b] * mask[b]
        current_run = 0
        for pos in range(seq_len):
            if float(example_errors[pos]) > 0:
                current_run += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
        if current_run > 0:
            run_lengths.append(current_run)

    if run_lengths:
        avg_run = sum(run_lengths) / len(run_lengths)
        max_run = max(run_lengths)
        single_bit = sum(1 for r in run_lengths if r == 1)
        multi_bit = sum(1 for r in run_lengths if r > 1)
    else:
        avg_run = max_run = 0
        single_bit = multi_bit = 0

    results = {
        "total_bit_errors": n_total_errors,
        "new_error_origins": n_new_errors,
        "propagated_errors": n_propagated,
        "propagation_ratio": n_propagated / max(n_total_errors, 1),
        "avg_error_run_length": avg_run,
        "max_error_run_length": max_run,
        "single_bit_error_runs": single_bit,
        "multi_bit_error_runs": multi_bit,
    }

    # If inputs are available, check correlation with input patterns
    if input_a is not None and input_b is not None:
        # Check if errors correlate with the number of 1-bits in inputs
        # (more 1s = more carries)
        a_ones = jnp.sum(input_a, axis=-1).astype(jnp.float32)
        b_ones = jnp.sum(input_b, axis=-1).astype(jnp.float32)
        example_errors = jnp.sum(errors, axis=-1)

        # Simple correlation
        total_ones = a_ones + b_ones
        corr = _pearson_correlation(total_ones, example_errors)
        results["input_ones_error_correlation"] = float(corr)

    return results


def _pearson_correlation(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute Pearson correlation between two 1D arrays."""
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numer = jnp.sum(x_centered * y_centered)
    denom = jnp.sqrt(jnp.sum(x_centered ** 2) * jnp.sum(y_centered ** 2))
    return numer / jnp.maximum(denom, 1e-8)


# ---------------------------------------------------------------------------
# Loss per recursion level
# ---------------------------------------------------------------------------

def loss_per_recursion_level(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    recursion_levels: jnp.ndarray,
) -> Dict[int, float]:
    """Compute cross-entropy loss broken down by recursion level.

    This shows whether the model struggles more at deeper recursion levels,
    which would indicate difficulty with the recursive structure.

    Args:
        logits: (batch, seq_len, vocab_size) model output logits.
        targets: (batch, seq_len) target token ids.
        mask: (batch, seq_len) output mask.
        recursion_levels: (batch, seq_len) recursion level for each token
            (0 = top level, higher = deeper recursion).

    Returns:
        Dict mapping recursion level (int) -> average cross-entropy loss.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)
    token_losses = -target_log_probs  # (batch, seq_len)

    max_level = int(jnp.max(recursion_levels))
    level_losses = {}

    for level in range(max_level + 1):
        level_mask = (recursion_levels == level).astype(jnp.float32) * mask
        level_total = jnp.sum(level_mask)
        if float(level_total) > 0:
            level_loss = jnp.sum(token_losses * level_mask) / level_total
            level_losses[level] = float(level_loss)
        else:
            level_losses[level] = 0.0

    return level_losses


# ---------------------------------------------------------------------------
# Combined metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    logits: Optional[jnp.ndarray] = None,
    recursion_levels: Optional[jnp.ndarray] = None,
    bit_significance: Optional[jnp.ndarray] = None,
    input_a: Optional[jnp.ndarray] = None,
    input_b: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Compute all available metrics in one call.

    Args:
        predictions: (batch, seq_len) predicted tokens.
        targets: (batch, seq_len) target tokens.
        mask: (batch, seq_len) output mask.
        logits: (batch, seq_len, vocab) model logits (for loss computation).
        recursion_levels: (batch, seq_len) recursion level ids.
        bit_significance: (batch, seq_len) bit position indices.
        input_a: (batch, input_len) first operand.
        input_b: (batch, input_len) second operand.

    Returns:
        Dict with all computed metrics.
    """
    results = {}

    # Token accuracy
    results["token_accuracy"] = token_accuracy(predictions, targets, mask)

    # Exact match
    results["exact_match"] = sequence_exact_match(predictions, targets, mask)

    # Per-bit-position accuracy
    bit_acc = per_bit_position_accuracy(
        predictions, targets, mask, bit_significance
    )
    results["per_bit_position"] = bit_acc

    # Carry propagation errors
    carry_errors = carry_propagation_errors(
        predictions, targets, mask, input_a, input_b
    )
    results["carry_errors"] = carry_errors

    # Loss per recursion level
    if logits is not None and recursion_levels is not None:
        level_losses = loss_per_recursion_level(
            logits, targets, mask, recursion_levels
        )
        results["loss_per_level"] = level_losses

    return results


# ---------------------------------------------------------------------------
# Metric aggregation across batches
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """Accumulate metrics over multiple batches and compute final averages.

    Usage:
        acc = MetricsAccumulator()
        for batch in batches:
            metrics = compute_all_metrics(...)
            acc.update(metrics, batch_size=len(batch))
        final = acc.compute()
    """

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, Any], batch_size: int = 1):
        """Add a batch of metrics."""
        self._update_recursive(metrics, batch_size, prefix="")

    def _update_recursive(self, d: Dict[str, Any], n: int, prefix: str):
        for k, v in d.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                self._update_recursive(v, n, full_key)
            elif isinstance(v, (int, float)):
                self._sums[full_key] = self._sums.get(full_key, 0.0) + v * n
                self._counts[full_key] = self._counts.get(full_key, 0) + n
            elif isinstance(v, list):
                # Store lists as-is (e.g. per_position accuracy)
                pass

    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self):
        """Clear accumulated metrics."""
        self._sums.clear()
        self._counts.clear()
