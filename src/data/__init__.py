"""
Data generation pipeline for Karatsuba-style recursive multiplication traces.

This package provides:
- karatsuba_trace: Karatsuba recursion trace generator (depth-first and breadth-first)
- school_trace: Grade-school / long multiplication trace generator (baseline)
- tokenizer: Tokenization and hierarchical position encoding
- dataset: JAX-compatible dataset with curriculum learning support
"""

from src.data.karatsuba_trace import KaratsubaTraceGenerator
from src.data.school_trace import SchoolTraceGenerator
from src.data.tokenizer import Tokenizer
from src.data.dataset import MultiplicationDataset
