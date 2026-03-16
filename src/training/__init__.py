"""
Training pipeline for Karatsuba looped transformer.

This package provides:
- train: JIT-compiled JAX/Equinox/Optax training loop with mixed precision,
         gradient accumulation, progressive loops, intermediate supervision,
         checkpointing, and W&B logging.
- evaluate: Length generalization evaluation (exact-match, per-digit, per-level accuracy)
            with autoregressive generation and JSON result export.
- curriculum: Curriculum learning with staged difficulty increases and
              mixed-difficulty batches.
"""

from src.training.train import (
    TrainState,
    create_train_state,
    train_step,
    train,
)
from src.training.evaluate import (
    evaluate_model,
    evaluate_length_generalization,
    autoregressive_generate,
)
from src.training.curriculum import (
    CurriculumScheduler,
    CurriculumStage,
)

__all__ = [
    # Training
    "TrainState",
    "create_train_state",
    "train_step",
    "train",
    # Evaluation
    "evaluate_model",
    "evaluate_length_generalization",
    "autoregressive_generate",
    # Curriculum
    "CurriculumScheduler",
    "CurriculumStage",
]
