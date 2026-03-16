"""
Analysis tools for Karatsuba looped transformer.

This package provides:
- metrics: Token-level and sequence-level accuracy, per-bit-position accuracy,
           carry propagation error analysis, loss per recursion level.
- attention_viz: Attention pattern visualization per head per loop iteration,
                 with input bit highlighting and cross-iteration comparison.
- mechanistic: Mechanistic interpretability tools including Fourier analysis
               of embeddings, residual stream analysis, ablation tools,
               and loop utilization analysis.
"""

from src.analysis.metrics import (
    token_accuracy,
    sequence_exact_match,
    per_bit_position_accuracy,
    carry_propagation_errors,
    loss_per_recursion_level,
)
from src.analysis.attention_viz import (
    plot_attention_weights,
    plot_attention_across_iterations,
    highlight_bit_attention,
)
from src.analysis.mechanistic import (
    fourier_analysis_embeddings,
    residual_stream_analysis,
    ablation_study,
    loop_utilization_analysis,
)

__all__ = [
    # Metrics
    "token_accuracy",
    "sequence_exact_match",
    "per_bit_position_accuracy",
    "carry_propagation_errors",
    "loss_per_recursion_level",
    # Attention visualization
    "plot_attention_weights",
    "plot_attention_across_iterations",
    "highlight_bit_attention",
    # Mechanistic interpretability
    "fourier_analysis_embeddings",
    "residual_stream_analysis",
    "ablation_study",
    "loop_utilization_analysis",
]
