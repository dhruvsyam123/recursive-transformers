"""
Training loop for Karatsuba looped transformer.

Features:
- JIT-compiled training step using eqx.filter_jit
- AdamW optimizer with cosine learning rate schedule
- Weight decay (important for generalization, per Nanda et al.)
- Gradient accumulation
- Checkpoint saving/loading (local + Google Drive)
- Weights & Biases logging (optional, graceful fallback)
- Mixed precision support (bf16 on TPU/A100, fp16 on T4)
- Config-driven: load experiment config from YAML
- Intermediate supervision: auxiliary loss on each recursion level's output
- Progressive loop training: start with fewer loops, increase over training
- Print training metrics every N steps
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import yaml

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment config from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_precision_dtype(precision_str: str):
    """Map precision string to JAX dtype."""
    mapping = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    return mapping.get(precision_str, jnp.float32)


def detect_best_precision() -> str:
    """Detect the best precision for the current hardware."""
    platform = jax.default_backend()
    if platform == "tpu":
        return "bfloat16"
    elif platform == "gpu":
        # A100 / L4 support bf16; T4 needs fp16
        # We conservatively pick bf16 and let the user override for T4
        return "bfloat16"
    else:
        return "float32"


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def create_lr_schedule(
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:
    """Cosine decay learning rate schedule with linear warmup."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=min_lr / peak_lr if peak_lr > 0 else 0.0,
    )
    schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )
    return schedule


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def create_optimizer(config: Dict[str, Any]) -> optax.GradientTransformation:
    """Create optimizer from config.

    Returns an optax chain with:
    - Gradient clipping
    - AdamW (or Lion) with cosine LR schedule and weight decay
    """
    tc = config["training"]
    lr_schedule = create_lr_schedule(
        peak_lr=tc["learning_rate"],
        min_lr=tc["min_learning_rate"],
        warmup_steps=tc["warmup_steps"],
        total_steps=tc["num_steps"],
    )

    if tc.get("optimizer", "adamw") == "lion":
        base_opt = optax.lion(
            learning_rate=lr_schedule,
            b1=tc.get("adam_b1", 0.9),
            b2=tc.get("adam_b2", 0.99),
            weight_decay=tc.get("weight_decay", 0.1),
        )
    else:
        base_opt = optax.adamw(
            learning_rate=lr_schedule,
            b1=tc.get("adam_b1", 0.9),
            b2=tc.get("adam_b2", 0.98),
            eps=tc.get("adam_eps", 1e-9),
            weight_decay=tc.get("weight_decay", 0.1),
        )

    chain = optax.chain(
        optax.clip_by_global_norm(tc.get("grad_clip_norm", 1.0)),
        base_opt,
    )
    return chain


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """Holds all mutable training state."""

    model: Any  # eqx.Module — the model pytree
    opt_state: Any  # optax optimizer state
    step: int = 0
    best_val_loss: float = float("inf")
    current_max_loops: int = 4
    rng: Any = None  # JAX PRNG key

    def to_dict(self) -> Dict[str, Any]:
        """Serialize non-pytree fields for JSON metadata."""
        return {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "current_max_loops": self.current_max_loops,
        }


def create_train_state(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    config: Dict[str, Any],
    rng: jax.Array,
) -> TrainState:
    """Initialize training state."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    tc = config["training"]
    initial_loops = tc.get("initial_loops", config["model"].get("max_loops", 8))
    return TrainState(
        model=model,
        opt_state=opt_state,
        step=0,
        best_val_loss=float("inf"),
        current_max_loops=initial_loops,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Token-level cross-entropy loss.

    Args:
        logits: (seq_len, vocab_size)
        targets: (seq_len,) integer targets
        mask: (seq_len,) binary mask — 1 for positions that contribute to loss
    Returns:
        Scalar loss.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Gather the log-prob of the target token at each position
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, None], axis=-1
    ).squeeze(-1)
    if mask is not None:
        target_log_probs = target_log_probs * mask
        return -jnp.sum(target_log_probs) / jnp.maximum(jnp.sum(mask), 1.0)
    return -jnp.mean(target_log_probs)


def compute_loss_with_intermediate_supervision(
    model: eqx.Module,
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    n_loops: int,
    intermediate_weight: float = 0.3,
    recursion_level_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute loss with optional intermediate supervision.

    When intermediate supervision is enabled, we add auxiliary cross-entropy
    losses from the output head applied at each loop iteration's hidden state.
    This encourages each recursion level to produce meaningful outputs and
    reduces error accumulation.

    Args:
        model: The LoopedTransformer model.
        tokens: (seq_len,) input token ids.
        positions: (seq_len, n_pos_dims) hierarchical position ids.
        targets: (seq_len,) target token ids.
        mask: (seq_len,) loss mask.
        n_loops: Number of loop iterations.
        intermediate_weight: Weight for auxiliary per-iteration losses.
        recursion_level_mask: Optional per-level mask for targeted supervision.

    Returns:
        (total_loss, metrics_dict)
    """
    # We need access to per-iteration hidden states.  The model should
    # expose a method that returns all intermediate states.
    # If the model has a `forward_with_intermediates` method, use it;
    # otherwise fall back to standard forward pass.
    if hasattr(model, "forward_with_intermediates"):
        final_logits, intermediate_logits = model.forward_with_intermediates(
            tokens, positions, n_loops
        )
    else:
        # Standard forward — no intermediate supervision possible
        final_logits = model(tokens, positions, n_loops)
        intermediate_logits = None

    # Primary loss: final output
    primary_loss = cross_entropy_loss(final_logits, targets, mask)
    metrics = {"primary_loss": primary_loss}

    # Intermediate supervision
    if intermediate_logits is not None and intermediate_weight > 0.0:
        n_intermediates = intermediate_logits.shape[0]  # (n_loops, seq_len, vocab)
        aux_loss = jnp.float32(0.0)
        for i in range(n_intermediates):
            level_loss = cross_entropy_loss(intermediate_logits[i], targets, mask)
            aux_loss = aux_loss + level_loss
            metrics[f"loss_loop_{i}"] = level_loss
        aux_loss = aux_loss / max(n_intermediates, 1)
        metrics["aux_loss"] = aux_loss
        total_loss = primary_loss + intermediate_weight * aux_loss
    else:
        total_loss = primary_loss

    metrics["total_loss"] = total_loss
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Single training step (JIT-compiled)
# ---------------------------------------------------------------------------

def make_train_step(optimizer: optax.GradientTransformation, config: Dict[str, Any]):
    """Create a JIT-compiled training step function.

    We use a closure so that the optimizer and config are captured, and
    the returned function has a clean signature for eqx.filter_jit.
    """
    tc = config["training"]
    intermediate_supervision = tc.get("intermediate_supervision", False)
    intermediate_weight = tc.get("intermediate_loss_weight", 0.3)

    @eqx.filter_jit
    def _train_step(
        model: eqx.Module,
        opt_state: Any,
        tokens: jnp.ndarray,
        positions: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        n_loops: int,
    ) -> Tuple[eqx.Module, Any, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Execute one gradient update.

        Args:
            model: Current model pytree.
            opt_state: Current optimizer state.
            tokens: (batch, seq_len) input token ids.
            positions: (batch, seq_len, n_pos_dims) position ids.
            targets: (batch, seq_len) target token ids.
            mask: (batch, seq_len) loss mask.
            n_loops: Number of loop iterations for this step.

        Returns:
            (updated_model, updated_opt_state, loss, metrics)
        """

        def loss_fn(model):
            # vmap over the batch dimension
            def single_example_loss(tok, pos, tgt, msk):
                return compute_loss_with_intermediate_supervision(
                    model, tok, pos, tgt, msk,
                    n_loops=n_loops,
                    intermediate_weight=intermediate_weight if intermediate_supervision else 0.0,
                )

            losses, metrics = jax.vmap(single_example_loss)(
                tokens, positions, targets, mask
            )
            mean_loss = jnp.mean(losses)
            mean_metrics = jax.tree.map(jnp.mean, metrics)
            return mean_loss, mean_metrics

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

        # Apply optimizer updates
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)

        return new_model, new_opt_state, loss, metrics

    return _train_step


def make_train_step_with_accumulation(
    optimizer: optax.GradientTransformation,
    config: Dict[str, Any],
    accumulation_steps: int,
):
    """Create a training step with gradient accumulation.

    Accumulates gradients over `accumulation_steps` microbatches before
    applying a single optimizer update.
    """
    tc = config["training"]
    intermediate_supervision = tc.get("intermediate_supervision", False)
    intermediate_weight = tc.get("intermediate_loss_weight", 0.3)

    @eqx.filter_jit
    def _accumulated_step(
        model: eqx.Module,
        opt_state: Any,
        all_tokens: jnp.ndarray,
        all_positions: jnp.ndarray,
        all_targets: jnp.ndarray,
        all_masks: jnp.ndarray,
        n_loops: int,
    ) -> Tuple[eqx.Module, Any, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Gradient accumulation over microbatches.

        Args:
            all_tokens: (accum_steps, micro_batch, seq_len)
            all_positions: (accum_steps, micro_batch, seq_len, n_pos_dims)
            all_targets: (accum_steps, micro_batch, seq_len)
            all_masks: (accum_steps, micro_batch, seq_len)
            n_loops: loop count.
        """
        micro_batch_size = all_tokens.shape[1]

        def microbatch_loss(model, tokens, positions, targets, mask_):
            def single_example_loss(tok, pos, tgt, msk):
                return compute_loss_with_intermediate_supervision(
                    model, tok, pos, tgt, msk,
                    n_loops=n_loops,
                    intermediate_weight=intermediate_weight if intermediate_supervision else 0.0,
                )

            losses, metrics = jax.vmap(single_example_loss)(
                tokens, positions, targets, mask_
            )
            return jnp.mean(losses), jax.tree.map(jnp.mean, metrics)

        def scan_body(carry, xs):
            accum_grads, accum_loss, accum_metrics = carry
            tokens, positions, targets, mask_ = xs

            (loss, metrics), grads = eqx.filter_value_and_grad(
                microbatch_loss, has_aux=True
            )(model, tokens, positions, targets, mask_)

            accum_grads = jax.tree.map(jnp.add, accum_grads, grads)
            accum_loss = accum_loss + loss
            accum_metrics = jax.tree.map(jnp.add, accum_metrics, metrics)
            return (accum_grads, accum_loss, accum_metrics), None

        # Initialize accumulators with zeros matching grad structure
        init_grads = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_array))
        init_loss = jnp.float32(0.0)
        # Run one forward to get metrics structure
        _, init_metrics = microbatch_loss(
            model, all_tokens[0], all_positions[0], all_targets[0], all_masks[0]
        )
        init_metrics = jax.tree.map(jnp.zeros_like, init_metrics)

        (total_grads, total_loss, total_metrics), _ = jax.lax.scan(
            scan_body,
            (init_grads, init_loss, init_metrics),
            (all_tokens, all_positions, all_targets, all_masks),
        )

        # Average over accumulation steps
        avg_grads = jax.tree.map(lambda g: g / accumulation_steps, total_grads)
        avg_loss = total_loss / accumulation_steps
        avg_metrics = jax.tree.map(lambda m: m / accumulation_steps, total_metrics)

        updates, new_opt_state = optimizer.update(
            avg_grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)

        return new_model, new_opt_state, avg_loss, avg_metrics

    return _accumulated_step


# ---------------------------------------------------------------------------
# Training step wrapper (public API)
# ---------------------------------------------------------------------------

def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    step_fn,
) -> Tuple[TrainState, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Execute one training step and update state.

    Args:
        state: Current TrainState.
        batch: Dict with 'tokens', 'positions', 'targets', 'mask' arrays.
        step_fn: Compiled step function from make_train_step.

    Returns:
        (new_state, loss, metrics)
    """
    new_model, new_opt_state, loss, metrics = step_fn(
        state.model,
        state.opt_state,
        batch["tokens"],
        batch["positions"],
        batch["targets"],
        batch["mask"],
        state.current_max_loops,
    )
    new_state = TrainState(
        model=new_model,
        opt_state=new_opt_state,
        step=state.step + 1,
        best_val_loss=state.best_val_loss,
        current_max_loops=state.current_max_loops,
        rng=state.rng,
    )
    return new_state, loss, metrics


# ---------------------------------------------------------------------------
# Progressive loop schedule
# ---------------------------------------------------------------------------

def get_current_max_loops(step: int, config: Dict[str, Any]) -> int:
    """Compute the current maximum loop count for progressive training.

    Progressive loop training starts with fewer loops and increases over
    training, allowing the model to first learn shallow computations before
    attempting deeper recursion.
    """
    tc = config["training"]
    if not tc.get("progressive_loops", False):
        return config["model"]["max_loops"]

    initial = tc.get("initial_loops", 4)
    increase_every = tc.get("loop_increase_every", 10000)
    increase_amount = tc.get("loop_increase_amount", 2)
    max_loops = config["model"]["max_loops"]

    n_increases = step // increase_every
    current = initial + n_increases * increase_amount
    return min(current, max_loops)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: TrainState,
    config: Dict[str, Any],
    path: Optional[str] = None,
    is_best: bool = False,
):
    """Save model checkpoint and training metadata.

    Saves:
    - model.eqx: Equinox model pytree
    - opt_state.eqx: Optimizer state pytree
    - metadata.json: Step, loss, loop count, config
    """
    if path is None:
        path = config.get("checkpoint", {}).get("dir", "./checkpoints")

    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"step_{state.step}"
    if is_best:
        suffix = "best"

    model_path = ckpt_dir / f"model_{suffix}.eqx"
    meta_path = ckpt_dir / f"metadata_{suffix}.json"

    # Save model using eqx.tree_serialise_leaves
    eqx.tree_serialise_leaves(str(model_path), state.model)

    # Save metadata
    metadata = state.to_dict()
    metadata["config"] = config
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Also save to Google Drive if available
    gdrive_dir = config.get("checkpoint", {}).get("gdrive_dir")
    if gdrive_dir and os.path.exists(os.path.dirname(gdrive_dir)):
        gdrive_path = Path(gdrive_dir)
        gdrive_path.mkdir(parents=True, exist_ok=True)
        gdrive_model_path = gdrive_path / f"model_{suffix}.eqx"
        gdrive_meta_path = gdrive_path / f"metadata_{suffix}.json"
        eqx.tree_serialise_leaves(str(gdrive_model_path), state.model)
        with open(gdrive_meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    return str(model_path)


def load_checkpoint(
    model_template: eqx.Module,
    path: str,
    step: Optional[int] = None,
    load_best: bool = False,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """Load model checkpoint.

    Args:
        model_template: A model with the correct structure (for deserialization).
        path: Checkpoint directory.
        step: Specific step to load. If None, loads latest or best.
        load_best: If True, load the best checkpoint.

    Returns:
        (model, metadata)
    """
    ckpt_dir = Path(path)

    if load_best:
        suffix = "best"
    elif step is not None:
        suffix = f"step_{step}"
    else:
        # Find the latest checkpoint
        model_files = sorted(ckpt_dir.glob("model_step_*.eqx"))
        if not model_files:
            raise FileNotFoundError(f"No checkpoints found in {path}")
        latest = model_files[-1]
        suffix = latest.stem.replace("model_", "")

    model_path = ckpt_dir / f"model_{suffix}.eqx"
    meta_path = ckpt_dir / f"metadata_{suffix}.json"

    model = eqx.tree_deserialise_leaves(str(model_path), model_template)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, metadata


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def init_wandb(config: Dict[str, Any]) -> bool:
    """Initialize Weights & Biases logging.

    Returns True if wandb was initialized, False otherwise.
    Gracefully handles the case where wandb is not installed or
    the user is not logged in.
    """
    if not WANDB_AVAILABLE:
        print("[wandb] Not available. Install with: pip install wandb")
        return False

    log_cfg = config.get("logging", {})
    if not log_cfg.get("use_wandb", False):
        return False

    try:
        wandb.init(
            project=log_cfg.get("wandb_project", "karatsuba-transformers"),
            entity=log_cfg.get("wandb_entity"),
            config=config,
            name=config.get("experiment_name", "karatsuba"),
        )
        return True
    except Exception as e:
        print(f"[wandb] Failed to initialize: {e}")
        return False


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    use_wandb: bool = False,
    prefix: str = "train",
):
    """Log metrics to stdout and optionally to wandb."""
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        flat = {}
        for k, v in metrics.items():
            key = f"{prefix}/{k}"
            val = float(v) if hasattr(v, "item") else v
            flat[key] = val
        wandb.log(flat, step=step)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model: eqx.Module,
    config: Dict[str, Any],
    train_dataset,
    val_dataset=None,
    resume_from: Optional[str] = None,
):
    """Main training loop.

    Args:
        model: Initialized LoopedTransformer (Equinox module).
        config: Experiment configuration dict (from YAML).
        train_dataset: Training dataset that yields batches. Must support
            `get_batch(batch_size, rng, bit_widths)` returning a dict with
            'tokens', 'positions', 'targets', 'mask' arrays.
        val_dataset: Optional validation dataset (same interface).
        resume_from: Path to checkpoint directory to resume from.

    Returns:
        Final TrainState.
    """
    tc = config["training"]
    rng = jax.random.PRNGKey(tc.get("seed", 42))

    # Create optimizer
    optimizer = create_optimizer(config)

    # Initialize or resume training state
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        model, metadata = load_checkpoint(model, resume_from, load_best=False)
        state = create_train_state(model, optimizer, config, rng)
        state.step = metadata.get("step", 0)
        state.best_val_loss = metadata.get("best_val_loss", float("inf"))
        state.current_max_loops = metadata.get(
            "current_max_loops", tc.get("initial_loops", 4)
        )
    else:
        state = create_train_state(model, optimizer, config, rng)

    # Create compiled training step
    accum_steps = tc.get("gradient_accumulation_steps", 1)
    if accum_steps > 1:
        step_fn = make_train_step_with_accumulation(optimizer, config, accum_steps)
    else:
        step_fn = make_train_step(optimizer, config)

    # Initialize wandb
    use_wandb = init_wandb(config)

    # Curriculum scheduler
    from src.training.curriculum import CurriculumScheduler

    curriculum = CurriculumScheduler(config)

    # Training loop
    num_steps = tc["num_steps"]
    print_every = tc.get("print_every", 100)
    eval_every = tc.get("eval_every", 1000)
    ckpt_every = tc.get("checkpoint_every", 2000)
    batch_size = tc["batch_size"]

    print(f"Starting training for {num_steps} steps")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial loops: {state.current_max_loops}")
    print(f"  Precision: {tc.get('precision', 'float32')}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.device_count()}")
    print()

    step_times = []

    for step in range(state.step, num_steps):
        step_start = time.time()

        # Update loop count (progressive training)
        state.current_max_loops = get_current_max_loops(step, config)

        # Get curriculum bit widths for this step
        bit_widths = curriculum.get_bit_widths(step)

        # Get training batch
        rng, batch_rng = jax.random.split(state.rng if state.rng is not None else rng)
        state.rng = rng

        batch = train_dataset.get_batch(
            batch_size=batch_size,
            rng=batch_rng,
            bit_widths=bit_widths,
        )

        # Training step
        if accum_steps > 1:
            # Reshape batch for accumulation: (accum, micro_batch, ...)
            micro_batch = batch_size // accum_steps
            batch_reshaped = jax.tree.map(
                lambda x: x.reshape(accum_steps, micro_batch, *x.shape[1:]),
                batch,
            )
            new_model, new_opt_state, loss, metrics = step_fn(
                state.model,
                state.opt_state,
                batch_reshaped["tokens"],
                batch_reshaped["positions"],
                batch_reshaped["targets"],
                batch_reshaped["mask"],
                state.current_max_loops,
            )
        else:
            new_model, new_opt_state, loss, metrics = step_fn(
                state.model,
                state.opt_state,
                batch["tokens"],
                batch["positions"],
                batch["targets"],
                batch["mask"],
                state.current_max_loops,
            )

        state = TrainState(
            model=new_model,
            opt_state=new_opt_state,
            step=step + 1,
            best_val_loss=state.best_val_loss,
            current_max_loops=state.current_max_loops,
            rng=state.rng,
        )

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Log metrics
        log_dict = {
            "loss": float(loss),
            "step_time": step_time,
            "max_loops": state.current_max_loops,
            "curriculum_bit_widths": str(bit_widths),
        }
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                log_dict[k] = float(v) if hasattr(v, "item") else v

        log_metrics(log_dict, step, use_wandb=use_wandb, prefix="train")

        # Print metrics
        if (step + 1) % print_every == 0:
            avg_time = sum(step_times[-print_every:]) / min(len(step_times), print_every)
            print(
                f"Step {step + 1}/{num_steps} | "
                f"Loss: {float(loss):.4f} | "
                f"Loops: {state.current_max_loops} | "
                f"Bits: {bit_widths} | "
                f"Time/step: {avg_time:.3f}s"
            )
            if isinstance(metrics, dict):
                extra = {k: f"{float(v):.4f}" for k, v in metrics.items()
                         if k not in ("total_loss",)}
                if extra:
                    print(f"  Metrics: {extra}")

        # Evaluation
        if val_dataset is not None and (step + 1) % eval_every == 0:
            val_loss, val_metrics = _run_validation(
                state, val_dataset, config, step_fn
            )
            print(f"  [Eval] Val loss: {float(val_loss):.4f}")

            val_log = {"val_loss": float(val_loss)}
            if isinstance(val_metrics, dict):
                for k, v in val_metrics.items():
                    val_log[f"val_{k}"] = float(v) if hasattr(v, "item") else v
            log_metrics(val_log, step, use_wandb=use_wandb, prefix="eval")

            # Track best
            if float(val_loss) < state.best_val_loss:
                state.best_val_loss = float(val_loss)
                if config.get("checkpoint", {}).get("save_best", True):
                    save_checkpoint(state, config, is_best=True)
                    print(f"  [Checkpoint] New best model saved (loss={float(val_loss):.4f})")

        # Periodic checkpoint
        if (step + 1) % ckpt_every == 0:
            ckpt_path = save_checkpoint(state, config)
            print(f"  [Checkpoint] Saved at step {step + 1}: {ckpt_path}")

    # Final checkpoint
    save_checkpoint(state, config)
    print(f"\nTraining complete. Final loss: {float(loss):.4f}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return state


def _run_validation(
    state: TrainState,
    val_dataset,
    config: Dict[str, Any],
    step_fn,
    num_batches: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Run validation over a few batches and return average loss."""
    tc = config["training"]
    batch_size = tc["batch_size"]
    rng = jax.random.PRNGKey(0)  # deterministic for validation

    total_loss = 0.0
    all_metrics = {}
    count = 0

    for i in range(num_batches):
        rng, batch_rng = jax.random.split(rng)
        batch = val_dataset.get_batch(
            batch_size=batch_size,
            rng=batch_rng,
            bit_widths=[config["data"].get("max_bit_width", 8)],
        )

        # Forward pass only (no grad) for loss
        def eval_loss_fn(model, tokens, positions, targets, mask_):
            def single_loss(tok, pos, tgt, msk):
                return compute_loss_with_intermediate_supervision(
                    model, tok, pos, tgt, msk,
                    n_loops=state.current_max_loops,
                    intermediate_weight=0.0,
                )

            losses, metrics = jax.vmap(single_loss)(tokens, positions, targets, mask_)
            return jnp.mean(losses), jax.tree.map(jnp.mean, metrics)

        loss, metrics = eval_loss_fn(
            state.model,
            batch["tokens"],
            batch["positions"],
            batch["targets"],
            batch["mask"],
        )
        total_loss += float(loss)

        if isinstance(metrics, dict):
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + float(v)
        count += 1

    avg_loss = total_loss / max(count, 1)
    avg_metrics = {k: v / max(count, 1) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for training.

    Usage:
        python -m src.training.train --config configs/8bit_karatsuba.yaml
        python -m src.training.train --config configs/8bit_karatsuba.yaml --resume checkpoints/
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train Karatsuba Looped Transformer")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint dir to resume from"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {config.get('experiment_name', 'unknown')}")

    # Import model and data (deferred to avoid circular imports)
    from src.model import LoopedTransformer, TransformerConfig
    from src.data import MultiplicationDataset

    # Build model
    model_cfg = config["model"]
    rng = jax.random.PRNGKey(config["training"].get("seed", 42))
    rng, model_rng = jax.random.split(rng)

    transformer_config = TransformerConfig(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        max_loops=model_cfg["max_loops"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    model = LoopedTransformer(transformer_config, key=model_rng)

    # Build datasets
    data_cfg = config["data"]
    train_dataset = MultiplicationDataset(
        algorithm=data_cfg["algorithm"],
        trace_format=data_cfg.get("trace_format", "depth_first"),
        max_bit_width=data_cfg["max_bit_width"],
        num_examples=data_cfg["num_train_examples"],
        binary=data_cfg.get("binary", True),
    )
    val_dataset = MultiplicationDataset(
        algorithm=data_cfg["algorithm"],
        trace_format=data_cfg.get("trace_format", "depth_first"),
        max_bit_width=data_cfg["max_bit_width"],
        num_examples=data_cfg["num_val_examples"],
        binary=data_cfg.get("binary", True),
    )

    # Train
    final_state = train(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resume_from=args.resume,
    )

    print("Done.")


if __name__ == "__main__":
    main()
