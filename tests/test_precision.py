"""
Tests for Flash Attention precision modes
=======================================

Tests FP16/FP32 precision handling and numerical accuracy.
"""

import pytest
import torch
import numpy as np
from flash_attention import flash_attention_forward, estimate_memory_usage


class TestPrecision:
    """Test suite for precision handling"""

    @pytest.mark.parametrize("use_half", [False, True])
    def test_precision_modes(self, use_half):
        """Test both FP32 and FP16 precision modes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 64

        dtype = torch.float16 if use_half else torch.float32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()

        with torch.no_grad():
            result = flash_attention_forward(q, k, v, use_half=use_half)

        assert result.dtype == dtype, f"Output dtype {result.dtype} doesn't match expected {dtype}"
        assert result.shape == q.shape, f"Output shape {result.shape} doesn't match input {q.shape}"
        assert torch.isfinite(result).all(), f"Result contains NaN/Inf for {'FP16' if use_half else 'FP32'}"

    def test_fp16_numerical_stability(self):
        """Test numerical stability with FP16"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

        # Create tensors that might cause numerical issues
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()

        # Scale to avoid overflow
        q = q * 0.1
        k = k * 0.1
        v = v * 0.1

        with torch.no_grad():
            result = flash_attention_forward(q, k, v, use_half=True)

        assert torch.isfinite(result).all(), "FP16 result contains NaN/Inf"
        assert result.abs().max() < 100.0, f"FP16 result magnitude too large: {result.abs().max()}"

    def test_precision_consistency(self):
        """Test consistency between FP32 and FP16 results"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

        # Create FP32 tensors
        q_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
        k_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
        v_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()

        # Convert to FP16
        q_fp16 = q_fp32.half()
        k_fp16 = k_fp32.half()
        v_fp16 = v_fp32.half()

        with torch.no_grad():
            result_fp32 = flash_attention_forward(q_fp32, k_fp32, v_fp32, use_half=False)
            result_fp16 = flash_attention_forward(q_fp16, k_fp16, v_fp16, use_half=True)

        # Convert FP16 result back to FP32 for comparison
        result_fp16_as_fp32 = result_fp16.float()

        # They should be reasonably close (allowing for FP16 precision limitations)
        assert torch.allclose(result_fp32, result_fp16_as_fp32, rtol=1e-2, atol=1e-2), \
            "FP32 and FP16 results differ too much"

    def test_mixed_precision_training(self):
        """Test mixed precision training setup"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 32

        # Create FP16 tensors for forward pass
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, requires_grad=True).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, requires_grad=True).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, requires_grad=True).cuda()

        # Forward pass in FP16
        out = flash_attention_forward(q, k, v, use_half=True)

        # Compute loss
        loss = out.sum()

        # Backward pass (gradients computed in FP16)
        loss.backward()

        # Check gradients
        assert q.grad is not None and torch.isfinite(q.grad).all(), "Query gradients invalid in mixed precision"
        assert k.grad is not None and torch.isfinite(k.grad).all(), "Key gradients invalid in mixed precision"
        assert v.grad is not None and torch.isfinite(v.grad).all(), "Value gradients invalid in mixed precision"

        assert q.grad.dtype == torch.float16, "Query gradient dtype should be FP16"
        assert k.grad.dtype == torch.float16, "Key gradient dtype should be FP16"
        assert v.grad.dtype == torch.float16, "Value gradient dtype should be FP16"

    def test_memory_usage_estimation(self):
        """Test memory usage estimation function"""
        test_cases = [
            # (batch_size, num_heads, seq_len, head_dim, use_half, expected_mb_range)
            (1, 1, 16, 32, False, (0.001, 0.01)),    # Small: ~0.001-0.01 MB
            (2, 8, 64, 64, False, (0.1, 1.0)),      # Medium: ~0.1-1.0 MB
            (4, 16, 128, 128, False, (5.0, 20.0)),  # Large: ~5-20 MB
            (1, 1, 16, 32, True, (0.0005, 0.005)),  # Small FP16: ~0.0005-0.005 MB
        ]

        for batch_size, num_heads, seq_len, head_dim, use_half, (min_mb, max_mb) in test_cases:
            mem_info = estimate_memory_usage(batch_size, num_heads, seq_len, head_dim, use_half)

            assert 'total_memory_mb' in mem_info, "Memory estimation missing total_memory_mb"
            total_mb = mem_info['total_memory_mb']

            assert min_mb <= total_mb <= max_mb, \
                f"Memory estimate {total_mb:.3f} MB not in expected range [{min_mb}, {max_mb}] for config " \
                f"B{batch_size}H{num_heads}N{seq_len}D{head_dim}{' FP16' if use_half else ' FP32'}"

            # Check that all memory components are present
            required_keys = ['input_memory', 'internal_memory', 'output_memory', 'total_memory']
            for key in required_keys:
                assert key in mem_info, f"Missing memory component: {key}"
                assert mem_info[key] > 0, f"Memory component {key} should be positive"

    def test_precision_edge_cases(self):
        """Test precision handling for edge cases"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 1, 8, 16

        # Test with very small values
        q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda() * 1e-6
        k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda() * 1e-6
        v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda() * 1e-6

        with torch.no_grad():
            result_fp32 = flash_attention_forward(q, k, v, use_half=False)
            result_fp16 = flash_attention_forward(q.half(), k.half(), v.half(), use_half=True)

        assert torch.isfinite(result_fp32).all(), "FP32 result contains NaN/Inf for small inputs"
        assert torch.isfinite(result_fp16).all(), "FP16 result contains NaN/Inf for small inputs"

        # Test with large values
        q_large = q * 1e6
        k_large = k * 1e6
        v_large = v * 1e6

        with torch.no_grad():
            result_fp32_large = flash_attention_forward(q_large, k_large, v_large, use_half=False)

        assert torch.isfinite(result_fp32_large).all(), "FP32 result contains NaN/Inf for large inputs"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_precision_determinism(self, dtype):
        """Test that results are deterministic for same inputs"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

        # Set seed for reproducibility
        torch.manual_seed(42)

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()

        use_half = (dtype == torch.float16)

        # Run multiple times
        results = []
        for _ in range(3):
            with torch.no_grad():
                result = flash_attention_forward(q, k, v, use_half=use_half)
            results.append(result.clone())

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], rtol=1e-5, atol=1e-6), \
                f"Results not deterministic between run 0 and run {i}"

    def test_automatic_precision_detection(self):
        """Test that precision mode is automatically detected from input tensors"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

        # Test FP32 tensors with use_half=False
        q_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
        k_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
        v_fp32 = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()

        with torch.no_grad():
            result_fp32 = flash_attention_forward(q_fp32, k_fp32, v_fp32, use_half=False)

        assert result_fp32.dtype == torch.float32, "FP32 input should produce FP32 output"

        # Test FP16 tensors with use_half=True
        q_fp16 = q_fp32.half()
        k_fp16 = k_fp32.half()
        v_fp16 = v_fp32.half()

        with torch.no_grad():
            result_fp16 = flash_attention_forward(q_fp16, k_fp16, v_fp16, use_half=True)

        assert result_fp16.dtype == torch.float16, "FP16 input should produce FP16 output"
