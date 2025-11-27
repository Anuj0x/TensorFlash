#!/usr/bin/env python3
"""
Basic Usage Examples for Modern Flash Attention
===============================================

This script demonstrates how to use the modernized Flash Attention implementation
with various configurations and use cases.
"""

import torch
import time
from flash_attention import (
    FlashAttention,
    flash_attention_forward,
    get_optimal_block_sizes,
    estimate_memory_usage
)


def basic_forward_pass():
    """Demonstrate basic forward pass usage"""
    print("=== Basic Forward Pass ===")

    # Define dimensions
    batch_size = 8
    num_heads = 16
    seq_len = 1024
    head_dim = 64

    print(f"Configuration: B={batch_size}, H={num_heads}, N={seq_len}, D={head_dim}")

    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()

    print(".1f")

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = flash_attention_forward(q, k, v)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    print(".4f")
    print(f"Output shape: {output.shape}")
    print()


def using_flash_attention_module():
    """Demonstrate using the FlashAttention module"""
    print("=== FlashAttention Module ===")

    # Create module with custom settings
    flash_attn = FlashAttention(bc=64, br=32, use_half=False)

    batch_size, num_heads, seq_len, head_dim = 4, 12, 512, 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()

    # Use as part of a model
    output = flash_attn(q, k, v)

    print(f"Module output shape: {output.shape}")
    print()


def half_precision_example():
    """Demonstrate half precision usage"""
    print("=== Half Precision (FP16) ===")

    batch_size, num_heads, seq_len, head_dim = 4, 8, 256, 64

    # Create FP16 tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()

    print(f"Input dtype: {q.dtype}")
    print(".1f")

    # Forward pass with FP16
    with torch.no_grad():
        output = flash_attention_forward(q, k, v, use_half=True)

    print(f"Output dtype: {output.dtype}")
    print(".1f")
    print()


def training_example():
    """Demonstrate training with backward pass"""
    print("=== Training Example (with Gradients) ===")

    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64

    # Create tensors with gradients enabled
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()

    print("Training tensors created with requires_grad=True")

    # Forward pass
    output = flash_attention_forward(q, k, v, use_half=False)
    loss = output.sum()

    print(f"Loss value: {loss.item():.4f}")

    # Backward pass
    start_time = time.time()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    print(".4f")
    print(f"Gradient shapes: Q={q.grad.shape}, K={k.grad.shape}, V={v.grad.shape}")
    print(f"All gradients finite: {torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()}")
    print()


def block_size_optimization():
    """Demonstrate block size optimization"""
    print("=== Block Size Optimization ===")

    seq_len, head_dim = 1024, 128

    # Get optimal block sizes
    bc_opt, br_opt = get_optimal_block_sizes(seq_len, head_dim)

    print(f"For sequence length {seq_len} and head dim {head_dim}:")
    print(f"Recommended Bc (columns): {bc_opt}")
    print(f"Recommended Br (rows): {br_opt}")

    # Test with different configurations
    configs = [(32, 32), (64, 32), (bc_opt, br_opt)]
    batch_size, num_heads = 2, 8

    for bc, br in configs:
        q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()

        start_time = time.time()
        with torch.no_grad():
            output = flash_attention_forward(q, k, v, bc=bc, br=br, use_half=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        print(".4f")

    print()


def memory_estimation_example():
    """Demonstrate memory usage estimation"""
    print("=== Memory Usage Estimation ===")

    test_configs = [
        (1, 1, 128, 64, False, "Small model"),
        (4, 12, 512, 64, False, "Medium model"),
        (8, 16, 1024, 128, True, "Large model (FP16)"),
    ]

    for batch_size, num_heads, seq_len, head_dim, use_half, desc in test_configs:
        mem_info = estimate_memory_usage(batch_size, num_heads, seq_len, head_dim, use_half)

        print(f"{desc}:")
        print(".1f")
        print(".1f")
        print()


def performance_comparison():
    """Compare performance with different configurations"""
    print("=== Performance Comparison ===")

    batch_size, num_heads, seq_len, head_dim = 4, 8, 256, 64

    configs = [
        ("FP32 Default", False, 32, 32),
        ("FP32 Large Blocks", False, 64, 64),
        ("FP16 Default", True, 32, 32),
        ("FP16 Large Blocks", True, 64, 64),
    ]

    for desc, use_half, bc, br in configs:
        q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.float16 if use_half else torch.float32).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.float16 if use_half else torch.float32).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.float16 if use_half else torch.float32).cuda()

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = flash_attention_forward(q, k, v, bc=bc, br=br, use_half=use_half)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        iterations = 10

        for _ in range(iterations):
            with torch.no_grad():
                output = flash_attention_forward(q, k, v, bc=bc, br=br, use_half=use_half)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        print(".4f")

    print()


def main():
    """Run all examples"""
    print("🚀 Modern Flash Attention - Usage Examples")
    print("=" * 50)
    print()

    if not torch.cuda.is_available():
        print("❌ CUDA not available. These examples require CUDA.")
        print("The code includes CPU fallback, but optimal performance requires GPU.")
        return

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    try:
        basic_forward_pass()
        using_flash_attention_module()
        half_precision_example()
        training_example()
        block_size_optimization()
        memory_estimation_example()
        performance_comparison()

        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
