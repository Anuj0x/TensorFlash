"""
Tests for Flash Attention forward pass
====================================

Tests correctness and performance of the forward pass implementation.
"""

import pytest
import torch
import numpy as np
import math
from flash_attention import flash_attention_forward, get_optimal_block_sizes


class TestForwardPass:
    """Test suite for Flash Attention forward pass"""

    def _reference_attention(self, q, k, v):
        """Reference implementation using standard attention"""
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("num_heads", [1, 8, 16])
    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_correctness_fp32(self, batch_size, num_heads, seq_len, head_dim):
        """Test correctness against reference implementation (FP32)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32).cuda()

        # Compute using Flash Attention
        with torch.no_grad():
            flash_result = flash_attention_forward(q, k, v, use_half=False)

        # Compute reference
        with torch.no_grad():
            ref_result = self._reference_attention(q, k, v)

        # Check numerical accuracy
        assert torch.allclose(flash_result, ref_result, rtol=1e-3, atol=1e-4), \
            "Flash Attention output doesn't match reference implementation"

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("num_heads", [1, 8, 16])
    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    @pytest.mark.parametrize("head_dim", [32, 64])
    def test_correctness_fp16(self, batch_size, num_heads, seq_len, head_dim):
        """Test correctness with half precision"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16).cuda()

        # Compute using Flash Attention
        with torch.no_grad():
            flash_result = flash_attention_forward(q, k, v, use_half=True)

        # Compute reference (convert to float for computation)
        with torch.no_grad():
            ref_result = self._reference_attention(q.float(), k.float(), v.float()).half()

        # Check numerical accuracy (more lenient for FP16)
        assert torch.allclose(flash_result, ref_result, rtol=1e-2, atol=1e-3), \
            "Flash Attention FP16 output doesn't match reference implementation"

    @pytest.mark.parametrize("bc,br", [(16, 16), (32, 32), (64, 32), (32, 64)])
    def test_block_sizes(self, bc, br):
        """Test different block size configurations"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()

        # Test with different block sizes
        with torch.no_grad():
            result1 = flash_attention_forward(q, k, v, bc=32, br=32, use_half=False)
            result2 = flash_attention_forward(q, k, v, bc=bc, br=br, use_half=False)

        # Results should be very close (within numerical precision)
        assert torch.allclose(result1, result2, rtol=1e-4, atol=1e-5), \
            f"Block size {bc}x{br} gives different results than 32x32"

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the forward pass"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()

        # Forward pass
        result = flash_attention_forward(q, k, v, use_half=False)
        loss = result.sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        assert q.grad is not None, "Query gradients not computed"
        assert k.grad is not None, "Key gradients not computed"
        assert v.grad is not None, "Value gradients not computed"

        # Check gradient shapes
        assert q.grad.shape == q.shape, f"Query gradient shape mismatch: {q.grad.shape} vs {q.shape}"
        assert k.grad.shape == k.shape, f"Key gradient shape mismatch: {k.grad.shape} vs {k.shape}"
        assert v.grad.shape == v.shape, f"Value gradient shape mismatch: {v.grad.shape} vs {v.shape}"

    def test_optimal_block_sizes(self):
        """Test the optimal block size selection utility"""
        test_cases = [
            (64, 32, (32, 32)),
            (128, 64, (32, 32)),
            (256, 64, (64, 32)),
            (512, 128, (64, 32)),
            (1024, 64, (128, 64)),
        ]

        for seq_len, head_dim, expected in test_cases:
            bc, br = get_optimal_block_sizes(seq_len, head_dim)
            assert bc <= head_dim, f"Block size {bc} exceeds head dimension {head_dim}"
            assert br <= head_dim, f"Block size {br} exceeds head dimension {head_dim}"
            assert bc > 0 and br > 0, "Block sizes must be positive"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_different_dtypes(self, dtype):
        """Test with different tensor data types"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype).cuda()

        use_half = (dtype == torch.float16)

        with torch.no_grad():
            result = flash_attention_forward(q, k, v, use_half=use_half)

        assert result.dtype == dtype, f"Output dtype {result.dtype} doesn't match input dtype {dtype}"
        assert result.shape == q.shape, f"Output shape {result.shape} doesn't match input shape {q.shape}"

    def test_cpu_fallback(self):
        """Test CPU fallback when CUDA is not available"""
        # This test will pass even without CUDA since we have CPU fallback
        batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Force CPU fallback by using CPU tensors
        result = flash_attention_forward(q, k, v, use_half=False)

        assert result.shape == q.shape, "CPU fallback output shape incorrect"
        assert result.device == q.device, "CPU fallback device mismatch"
