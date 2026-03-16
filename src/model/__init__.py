# Model package for Karatsuba Looped Transformer
#
# Architecture: JAX + Equinox looped transformer with:
# - Shared weight blocks looped via jax.lax.scan
# - Hierarchical position encodings for recursive structure
# - Adaptive computation time (ACT) halting mechanism
# - RMSNorm, pre-LayerNorm, timestep embeddings

from src.model.looped_transformer import (
    LoopedTransformer,
    LoopedTransformerBlock,
    TransformerConfig,
    RMSNorm,
)
from src.model.position_encoding import (
    SinusoidalPositionEncoding,
    HierarchicalPositionEncoding,
    PositionCoupling,
    LearnablePositionEncoding,
)
from src.model.halting import (
    HaltingMechanism,
    ACTState,
    adaptive_computation_time,
)

__all__ = [
    # Core model
    "LoopedTransformer",
    "LoopedTransformerBlock",
    "TransformerConfig",
    "RMSNorm",
    # Position encodings
    "SinusoidalPositionEncoding",
    "HierarchicalPositionEncoding",
    "PositionCoupling",
    "LearnablePositionEncoding",
    # Halting / ACT
    "HaltingMechanism",
    "ACTState",
    "adaptive_computation_time",
]
