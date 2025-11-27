"""
Python interface for Modern Flash Attention
==========================================

High-level Python interface with PyTorch integration.
"""

import torch
import math
from typing import Optional, Tuple, Union
from torch.utils.cpp_extension import load

# Load the CUDA extension
try:
    _flash_attn = load(
        name='minimal_attn',
        sources=['src/cpp/main.cpp', 'src/cuda/flash.cu'],
        extra_cuda_cflags=['-O2', '-arch=sm_80', '--use_fast_math'],
        extra_cflags=['-std=c++17'],
        verbose=False
    )
except Exception as e:
    print(f"Warning: Could not load CUDA extension: {e}")
    print("Using CPU fallback (limited functionality)")
    _flash_attn = None


class FlashAttention(torch.nn.Module):
    """
    Modern Flash Attention module for PyTorch.

    Supports both forward and backward passes with configurable parameters.

    Args:
        bc (int): Block size for columns (default: 32)
        br (int): Block size for rows (default: 32)
        use_half (bool): Use FP16 precision (default: False)
    """

    def __init__(self, bc: int = 32, br: int = 32, use_half: bool = False):
        super().__init__()
        self.bc = bc
        self.br = br
        self.use_half = use_half

        if _flash_attn is None:
            raise RuntimeError("CUDA extension not available. Please install CUDA and PyTorch with CUDA support.")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Flash Attention.

        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            v: Value tensor [B, H, N, D]

        Returns:
            Output tensor [B, H, N, D]
        """
        return flash_attention_forward(q, k, v, self.bc, self.br, self.use_half)

    def backward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 o: torch.Tensor, do: torch.Tensor, l: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass of Flash Attention.

        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            v: Value tensor [B, H, N, D]
            o: Output from forward pass
            do: Gradient w.r.t. output
            l: Row sums from forward pass
            m: Row maxima from forward pass

        Returns:
            Tuple of gradients (dQ, dK, dV)
        """
        return flash_attention_backward(q, k, v, o, do, l, m, self.bc, self.br, self.use_half)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bc: int = 32,
    br: int = 32,
    use_half: bool = False
) -> torch.Tensor:
    """
    Functional interface for Flash Attention forward pass.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        bc: Block size for columns
        br: Block size for rows
        use_half: Use FP16 precision

    Returns:
        Output tensor [B, H, N, D]
    """
    if _flash_attn is None:
        # CPU fallback implementation (basic attention)
        return _cpu_attention_forward(q, k, v)

    return _flash_attn.flash_attention_forward(q, k, v, bc, br, use_half)


def flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    bc: int = 32,
    br: int = 32,
    use_half: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Functional interface for Flash Attention backward pass.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        o: Output from forward pass
        do: Gradient w.r.t. output
        l: Row sums from forward pass
        m: Row maxima from forward pass
        bc: Block size for columns
        br: Block size for rows
        use_half: Use FP16 precision

    Returns:
        Tuple of gradients (dQ, dK, dV)
    """
    if _flash_attn is None:
        raise RuntimeError("Backward pass requires CUDA extension")

    grads = _flash_attn.flash_attention_backward(q, k, v, o, do, l, m, bc, br, use_half)
    return grads[0], grads[1], grads[2]


def _cpu_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    CPU fallback implementation for basic attention.
    Much slower than CUDA version but allows testing on CPU-only systems.
    """
    # Standard attention: softmax(QK^T / sqrt(d))V
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


# Utility functions
def get_optimal_block_sizes(sequence_length: int, head_dim: int) -> Tuple[int, int]:
    """
    Get optimal block sizes based on sequence length and head dimension.

    Args:
        sequence_length: Length of the sequence
        head_dim: Dimension of each attention head

    Returns:
        Tuple of (Bc, Br) block sizes
    """
    # Simple heuristics - can be improved with more sophisticated tuning
    if sequence_length <= 128:
        bc, br = 32, 32
    elif sequence_length <= 512:
        bc, br = 64, 32
    elif sequence_length <= 1024:
        bc, br = 128, 64
    else:
        bc, br = 256, 128

    # Ensure block sizes are reasonable for the head dimension
    bc = min(bc, head_dim)
    br = min(br, head_dim)

    return bc, br


def estimate_memory_usage(batch_size: int, num_heads: int, seq_len: int, head_dim: int,
                         use_half: bool = False) -> dict:
    """
    Estimate memory usage for Flash Attention.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        use_half: Whether using half precision

    Returns:
        Dictionary with memory estimates
    """
    element_size = 2 if use_half else 4  # bytes per element

    # Input tensors
    qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * element_size

    # Internal tensors (l, m) - always float32
    internal_memory = 2 * batch_size * num_heads * seq_len * 4

    # Output tensor
    output_memory = batch_size * num_heads * seq_len * head_dim * element_size

    total_memory = qkv_memory + internal_memory + output_memory

    return {
        'input_memory': qkv_memory,
        'internal_memory': internal_memory,
        'output_memory': output_memory,
        'total_memory': total_memory,
        'total_memory_mb': total_memory / (1024 * 1024)
    }


# Export public API
__all__ = [
    'FlashAttention',
    'flash_attention_forward',
    'flash_attention_backward',
    'get_optimal_block_sizes',
    'estimate_memory_usage'
]
