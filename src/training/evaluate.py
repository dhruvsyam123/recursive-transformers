"""
Evaluation and length generalization tests for the Karatsuba looped transformer.

Features:
- Exact-match accuracy at each test length (8, 16, 32, 64, 128 bits)
- Per-digit accuracy (to see where errors occur)
- Per-recursion-level accuracy
- Autoregressive generation from trained model
- Comparison metrics between Karatsuba and school algorithm
- Save evaluation results as JSON
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------

def autoregressive_generate(
    model: eqx.Module,
    input_tokens: jnp.ndarray,
    input_positions: jnp.ndarray,
    n_loops: int,
    max_new_tokens: int,
    eos_token_id: int = 5,
    temperature: float = 0.0,
    rng: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Generate tokens autoregressively from the model.

    Feeds the model the input prefix, then generates one token at a time
    by feeding the full sequence (input + generated so far) back through
    the model at each step.

    Args:
        model: Trained LoopedTransformer.
        input_tokens: (prefix_len,) input token ids.
        input_positions: (prefix_len, n_pos_dims) hierarchical position ids.
        n_loops: Number of loop iterations for the model.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: Token id for end-of-sequence (stop generation).
        temperature: Sampling temperature. 0.0 = greedy (argmax).
        rng: JAX PRNG key (required if temperature > 0).

    Returns:
        (total_len,) array of token ids (input + generated).
    """
    generated = list(input_tokens)
    positions = list(input_positions)
    prefix_len = len(input_tokens)

    for step in range(max_new_tokens):
        # Prepare current sequence as arrays
        tok_arr = jnp.array(generated, dtype=jnp.int32)
        pos_arr = jnp.array(positions)

        # Forward pass: get logits for all positions
        logits = model(tok_arr, pos_arr, n_loops)  # (seq_len, vocab_size)

        # We only care about the logits at the last position
        next_logits = logits[-1]  # (vocab_size,)

        if temperature <= 0.0:
            # Greedy decoding
            next_token = jnp.argmax(next_logits)
        else:
            # Temperature sampling
            assert rng is not None, "rng required for temperature > 0"
            rng, sample_rng = jax.random.split(rng)
            scaled_logits = next_logits / temperature
            next_token = jax.random.categorical(sample_rng, scaled_logits)

        next_token_int = int(next_token)
        generated.append(next_token_int)

        # Create position for new token (use a placeholder — the dataset
        # should provide a position-generation function for test lengths)
        if len(positions) > 0:
            # Simple heuristic: increment the first position dimension
            last_pos = positions[-1]
            if hasattr(last_pos, '__len__'):
                new_pos = list(last_pos)
                new_pos[0] = new_pos[0] + 1  # increment bit significance
                positions.append(new_pos)
            else:
                positions.append(last_pos + 1)

        # Check for EOS
        if next_token_int == eos_token_id:
            break

    return jnp.array(generated, dtype=jnp.int32)


@eqx.filter_jit
def _batch_forward(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    n_loops: int,
) -> jnp.ndarray:
    """JIT-compiled batched forward pass.

    Args:
        model: LoopedTransformer.
        tokens: (batch, seq_len) input tokens.
        positions: (batch, seq_len, n_pos_dims) positions.
        n_loops: Loop count.

    Returns:
        (batch, seq_len, vocab_size) logits.
    """
    return jax.vmap(lambda t, p: model(t, p, n_loops))(tokens, positions)


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def exact_match_accuracy(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
) -> float:
    """Compute exact-match (sequence-level) accuracy.

    A prediction is correct only if ALL masked positions match the target.

    Args:
        predictions: (batch, seq_len) predicted token ids.
        targets: (batch, seq_len) target token ids.
        mask: (batch, seq_len) binary mask (1 for output positions).

    Returns:
        Fraction of sequences where all masked positions are correct.
    """
    correct_tokens = (predictions == targets) | (mask == 0)
    all_correct = jnp.all(correct_tokens, axis=-1)  # (batch,)
    return float(jnp.mean(all_correct))


def per_digit_accuracy(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
) -> Dict[str, Any]:
    """Compute per-digit (per-bit-position) accuracy.

    Returns accuracy for each position in the output sequence, revealing
    where errors concentrate (e.g., carry propagation in MSBs).

    Args:
        predictions: (batch, seq_len) predicted token ids.
        targets: (batch, seq_len) target token ids.
        mask: (batch, seq_len) binary mask.

    Returns:
        Dict with 'per_position_accuracy' (list of floats) and
        'mean_token_accuracy' (float).
    """
    correct = (predictions == targets).astype(jnp.float32) * mask
    # Per-position accuracy (average over batch)
    position_correct = jnp.sum(correct, axis=0)  # (seq_len,)
    position_count = jnp.sum(mask, axis=0)  # (seq_len,)
    position_acc = jnp.where(
        position_count > 0,
        position_correct / position_count,
        jnp.float32(0.0),
    )

    # Overall token accuracy
    total_correct = jnp.sum(correct)
    total_count = jnp.sum(mask)
    mean_acc = float(total_correct / jnp.maximum(total_count, 1.0))

    return {
        "per_position_accuracy": [float(a) for a in position_acc],
        "mean_token_accuracy": mean_acc,
    }


def per_recursion_level_accuracy(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    level_ids: jnp.ndarray,
) -> Dict[int, float]:
    """Compute accuracy for each recursion level.

    Useful for diagnosing at which recursion depth the model starts failing.

    Args:
        predictions: (batch, seq_len) predicted token ids.
        targets: (batch, seq_len) target token ids.
        mask: (batch, seq_len) binary mask.
        level_ids: (batch, seq_len) recursion level for each token (0=top level).

    Returns:
        Dict mapping recursion level -> accuracy.
    """
    correct = (predictions == targets).astype(jnp.float32) * mask
    max_level = int(jnp.max(level_ids))
    level_accs = {}

    for level in range(max_level + 1):
        level_mask = (level_ids == level).astype(jnp.float32) * mask
        level_correct = jnp.sum(correct * (level_ids == level).astype(jnp.float32))
        level_total = jnp.sum(level_mask)
        if float(level_total) > 0:
            level_accs[level] = float(level_correct / level_total)
        else:
            level_accs[level] = 0.0

    return level_accs


# ---------------------------------------------------------------------------
# Evaluate model on a single bit width
# ---------------------------------------------------------------------------

def evaluate_model(
    model: eqx.Module,
    dataset,
    bit_width: int,
    n_loops: int,
    num_examples: int = 1024,
    batch_size: int = 128,
    rng: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """Evaluate model on a specific bit width.

    Args:
        model: Trained LoopedTransformer.
        dataset: Dataset that can generate batches at the given bit width.
        bit_width: Test bit width.
        n_loops: Number of loop iterations.
        num_examples: Total examples to evaluate.
        batch_size: Batch size for evaluation.
        rng: PRNG key.

    Returns:
        Dict with all evaluation metrics.
    """
    if rng is None:
        rng = jax.random.PRNGKey(12345)

    all_predictions = []
    all_targets = []
    all_masks = []
    all_level_ids = []
    total_loss = 0.0
    n_batches = 0

    num_batches = max(1, num_examples // batch_size)

    for i in range(num_batches):
        rng, batch_rng = jax.random.split(rng)
        batch = dataset.get_batch(
            batch_size=batch_size,
            rng=batch_rng,
            bit_widths=[bit_width],
        )

        tokens = batch["tokens"]      # (batch, seq_len)
        positions = batch["positions"] # (batch, seq_len, n_pos_dims)
        targets = batch["targets"]     # (batch, seq_len)
        mask = batch["mask"]           # (batch, seq_len)

        # Forward pass
        logits = _batch_forward(model, tokens, positions, n_loops)
        predictions = jnp.argmax(logits, axis=-1)  # (batch, seq_len)

        # Compute loss
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)
        batch_loss = -jnp.sum(target_log_probs * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        total_loss += float(batch_loss)

        all_predictions.append(predictions)
        all_targets.append(targets)
        all_masks.append(mask)
        if "recursion_level" in batch:
            all_level_ids.append(batch["recursion_level"])
        n_batches += 1

    # Concatenate all results
    all_predictions = jnp.concatenate(all_predictions, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    all_masks = jnp.concatenate(all_masks, axis=0)

    # Compute metrics
    em_acc = exact_match_accuracy(all_predictions, all_targets, all_masks)
    digit_acc = per_digit_accuracy(all_predictions, all_targets, all_masks)

    results = {
        "bit_width": bit_width,
        "n_loops": n_loops,
        "num_examples": int(all_predictions.shape[0]),
        "exact_match_accuracy": em_acc,
        "mean_token_accuracy": digit_acc["mean_token_accuracy"],
        "per_position_accuracy": digit_acc["per_position_accuracy"],
        "mean_loss": total_loss / max(n_batches, 1),
    }

    # Per-level accuracy if recursion level info is available
    if all_level_ids:
        all_level_ids = jnp.concatenate(all_level_ids, axis=0)
        level_acc = per_recursion_level_accuracy(
            all_predictions, all_targets, all_masks, all_level_ids
        )
        results["per_recursion_level_accuracy"] = level_acc

    return results


# ---------------------------------------------------------------------------
# Length generalization evaluation
# ---------------------------------------------------------------------------

def evaluate_length_generalization(
    model: eqx.Module,
    dataset,
    config: Dict[str, Any],
    rng: Optional[jax.Array] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate model at multiple test lengths for length generalization.

    Tests the model at each bit width in the evaluation config and
    computes exact-match, per-digit, and per-recursion-level accuracy.

    Args:
        model: Trained LoopedTransformer.
        dataset: Dataset supporting arbitrary bit widths.
        config: Experiment config dict.
        rng: PRNG key.
        save_path: If set, save results as JSON to this path.

    Returns:
        Dict mapping bit_width -> evaluation results.
    """
    if rng is None:
        rng = jax.random.PRNGKey(99999)

    eval_cfg = config.get("evaluation", {})
    test_lengths = eval_cfg.get("test_lengths",
                                eval_cfg.get("test_bit_widths", [8, 16, 32, 64, 128]))
    num_test = eval_cfg.get("n_eval_samples_per_length",
                            eval_cfg.get("num_test_examples", 1024))
    batch_size = config.get("training", {}).get("batch_size", 128)

    # Loop counts for each test length (more loops for deeper recursion)
    test_loop_counts = eval_cfg.get("test_loop_counts", {})
    default_loops = config.get("model", {}).get("max_loops", 8)

    all_results = {}

    print("=" * 60)
    print("Length Generalization Evaluation")
    print("=" * 60)

    for bit_width in test_lengths:
        rng, eval_rng = jax.random.split(rng)
        n_loops = test_loop_counts.get(str(bit_width),
                                       test_loop_counts.get(bit_width, default_loops))

        print(f"\n--- {bit_width}-bit multiplication (n_loops={n_loops}) ---")
        start_time = time.time()

        try:
            results = evaluate_model(
                model=model,
                dataset=dataset,
                bit_width=bit_width,
                n_loops=n_loops,
                num_examples=num_test,
                batch_size=min(batch_size, num_test),
                rng=eval_rng,
            )
            elapsed = time.time() - start_time

            print(f"  Exact match accuracy: {results['exact_match_accuracy']:.4f}")
            print(f"  Mean token accuracy:  {results['mean_token_accuracy']:.4f}")
            print(f"  Mean loss:            {results['mean_loss']:.4f}")
            if "per_recursion_level_accuracy" in results:
                print(f"  Per-level accuracy:   {results['per_recursion_level_accuracy']}")
            print(f"  Time: {elapsed:.1f}s")

            results["elapsed_seconds"] = elapsed
            all_results[bit_width] = results

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[bit_width] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Bit Width':>10} | {'Exact Match':>12} | {'Token Acc':>10} | {'Loss':>8}")
    print("-" * 50)
    for bw in test_lengths:
        r = all_results.get(bw, {})
        if "error" in r:
            print(f"{bw:>10} | {'ERROR':>12} | {'':>10} | {'':>8}")
        else:
            print(
                f"{bw:>10} | "
                f"{r.get('exact_match_accuracy', 0):.4f}       | "
                f"{r.get('mean_token_accuracy', 0):.4f}     | "
                f"{r.get('mean_loss', 0):.4f}"
            )

    # Save results
    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        # Convert all values to JSON-serializable types
        serializable = _make_serializable(all_results)
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {save_path}")

    return all_results


# ---------------------------------------------------------------------------
# Comparison between Karatsuba and school algorithm
# ---------------------------------------------------------------------------

def compare_algorithms(
    karatsuba_model: eqx.Module,
    school_model: eqx.Module,
    karatsuba_dataset,
    school_dataset,
    config: Dict[str, Any],
    test_bit_widths: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare Karatsuba vs school algorithm models.

    Evaluates both models at each test length and produces a
    side-by-side comparison of exact-match accuracy, token accuracy,
    and computational efficiency (number of steps/tokens required).

    Args:
        karatsuba_model: Model trained on Karatsuba traces.
        school_model: Model trained on school algorithm traces.
        karatsuba_dataset: Dataset for Karatsuba evaluation.
        school_dataset: Dataset for school algorithm evaluation.
        config: Experiment config.
        test_bit_widths: Bit widths to test. Uses config defaults if None.
        save_path: Path to save comparison results as JSON.

    Returns:
        Comparison results dict.
    """
    if test_bit_widths is None:
        eval_cfg = config.get("evaluation", {})
        test_bit_widths = eval_cfg.get("test_lengths",
                                       eval_cfg.get("test_bit_widths", [8, 16, 32, 64]))

    default_loops = config.get("model", {}).get("max_loops", 8)
    rng = jax.random.PRNGKey(777)

    comparison = {}
    print("=" * 70)
    print("Algorithm Comparison: Karatsuba vs School")
    print("=" * 70)

    for bw in test_bit_widths:
        rng, r1, r2 = jax.random.split(rng, 3)

        # Evaluate Karatsuba model
        k_results = evaluate_model(
            karatsuba_model, karatsuba_dataset, bw, default_loops, rng=r1
        )
        # Evaluate school model
        s_results = evaluate_model(
            school_model, school_dataset, bw, default_loops, rng=r2
        )

        comparison[bw] = {
            "karatsuba": {
                "exact_match": k_results["exact_match_accuracy"],
                "token_accuracy": k_results["mean_token_accuracy"],
                "loss": k_results["mean_loss"],
            },
            "school": {
                "exact_match": s_results["exact_match_accuracy"],
                "token_accuracy": s_results["mean_token_accuracy"],
                "loss": s_results["mean_loss"],
            },
            "delta_exact_match": (
                k_results["exact_match_accuracy"] - s_results["exact_match_accuracy"]
            ),
        }

        print(f"\n{bw}-bit multiplication:")
        print(f"  Karatsuba: EM={k_results['exact_match_accuracy']:.4f}, "
              f"Token={k_results['mean_token_accuracy']:.4f}")
        print(f"  School:    EM={s_results['exact_match_accuracy']:.4f}, "
              f"Token={s_results['mean_token_accuracy']:.4f}")
        delta = comparison[bw]["delta_exact_match"]
        winner = "Karatsuba" if delta > 0 else "School" if delta < 0 else "Tie"
        print(f"  Winner: {winner} (delta={delta:+.4f})")

    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        serializable = _make_serializable(comparison)
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nComparison saved to: {save_path}")

    return comparison


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_errors(
    model: eqx.Module,
    dataset,
    bit_width: int,
    n_loops: int,
    num_examples: int = 256,
    rng: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """Detailed error analysis for a specific bit width.

    Categorizes errors by type:
    - Off-by-one errors (carry propagation issues)
    - MSB errors (overflow)
    - LSB errors (base case mistakes)
    - Multi-bit errors
    - Recursion boundary errors

    Args:
        model: Trained model.
        dataset: Evaluation dataset.
        bit_width: Test bit width.
        n_loops: Loop iterations.
        num_examples: Number of examples to analyze.
        rng: PRNG key.

    Returns:
        Error analysis dict.
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)

    batch = dataset.get_batch(
        batch_size=num_examples,
        rng=rng,
        bit_widths=[bit_width],
    )

    tokens = batch["tokens"]
    positions = batch["positions"]
    targets = batch["targets"]
    mask = batch["mask"]

    logits = _batch_forward(model, tokens, positions, n_loops)
    predictions = jnp.argmax(logits, axis=-1)

    # Identify incorrect examples
    correct_tokens = (predictions == targets) | (mask == 0)
    all_correct = jnp.all(correct_tokens, axis=-1)
    incorrect_mask = ~all_correct

    num_incorrect = int(jnp.sum(incorrect_mask))
    num_total = int(all_correct.shape[0])

    # Analyze error positions
    error_positions = ((predictions != targets) & (mask == 1)).astype(jnp.float32)

    # Average error density per position
    if num_incorrect > 0:
        # Only look at incorrect examples
        incorrect_errors = error_positions * incorrect_mask[:, None].astype(jnp.float32)
        error_density = jnp.sum(incorrect_errors, axis=0) / max(num_incorrect, 1)
    else:
        error_density = jnp.zeros(error_positions.shape[1])

    # Count single-bit vs multi-bit errors
    errors_per_example = jnp.sum(error_positions, axis=-1)
    single_bit_errors = int(jnp.sum(errors_per_example == 1))
    multi_bit_errors = int(jnp.sum(errors_per_example > 1))

    # LSB vs MSB error analysis (first half vs second half of output)
    seq_len = mask.shape[1]
    mid = seq_len // 2
    lsb_errors = jnp.sum(error_positions[:, :mid])
    msb_errors = jnp.sum(error_positions[:, mid:])

    results = {
        "bit_width": bit_width,
        "total_examples": num_total,
        "incorrect_examples": num_incorrect,
        "error_rate": num_incorrect / max(num_total, 1),
        "single_bit_errors": single_bit_errors,
        "multi_bit_errors": multi_bit_errors,
        "lsb_error_fraction": float(lsb_errors / jnp.maximum(lsb_errors + msb_errors, 1)),
        "msb_error_fraction": float(msb_errors / jnp.maximum(lsb_errors + msb_errors, 1)),
        "error_density_per_position": [float(x) for x in error_density],
    }

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert JAX arrays and numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (jnp.ndarray,)):
        return obj.tolist()
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for evaluation.

    Usage:
        python -m src.training.evaluate \\
            --config configs/8bit_karatsuba.yaml \\
            --checkpoint checkpoints/best \\
            --output results/eval_results.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Karatsuba Transformer")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir")
    parser.add_argument("--output", type=str, default="results/eval_results.json",
                        help="Output JSON path")
    parser.add_argument("--best", action="store_true", help="Load best checkpoint")
    args = parser.parse_args()

    from src.training.train import load_config, load_checkpoint
    from src.model import LoopedTransformer, TransformerConfig
    from src.data import MultiplicationDataset

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    # Build model template
    model_cfg = config["model"]
    rng = jax.random.PRNGKey(0)
    transformer_config = TransformerConfig(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg.get("n_shared_layers", model_cfg.get("n_layers", 2)),
        d_ff=model_cfg["d_ff"],
        max_loops=model_cfg.get("max_loops", 8),
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    model_template = LoopedTransformer(transformer_config, key=rng)

    # Load checkpoint
    model, metadata = load_checkpoint(
        model_template, args.checkpoint, load_best=args.best
    )
    print(f"Loaded checkpoint from step {metadata.get('step', '?')}")

    # Build evaluation dataset
    data_cfg = config.get("data", {})
    dataset = MultiplicationDataset(
        algorithm=data_cfg.get("algorithm", "karatsuba"),
        trace_format=data_cfg.get("trace_format", "depth_first"),
        max_bit_width=max(config.get("evaluation", {}).get("test_lengths", [64])),
        num_examples=config.get("evaluation", {}).get("n_eval_samples_per_length", 1024),
        binary=data_cfg.get("binary", data_cfg.get("representation") == "binary"),
    )

    # Run evaluation
    results = evaluate_length_generalization(
        model=model,
        dataset=dataset,
        config=config,
        save_path=args.output,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
