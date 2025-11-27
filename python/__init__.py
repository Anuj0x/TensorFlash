"""
Modern Flash Attention - High-performance CUDA implementation
===========================================================

A modernized, production-ready implementation of Flash Attention with:
- Backward pass support for training
- Half precision (FP16) acceleration
- Dynamic block sizes
- Modern CUDA features (cooperative groups, async memory)
- Comprehensive testing and benchmarking

Author: Modernized from tspeterkim/flash-attention-minimal
License: Apache 2.0
"""

from .flash_attention import (
    flash_attention_forward,
    flash_attention_backward,
    FlashAttention
)

__version__ = "1.0.0"
__author__ = "Modern Flash Attention Team"
__license__ = "Apache 2.0"

__all__ = [
    "flash_attention_forward",
    "flash_attention_backward",
    "FlashAttention",
    "__version__",
    "__author__",
    "__license__"
]
