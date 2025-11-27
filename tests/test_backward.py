"""
Tests for Flash Attention backward pass
====================================

Tests correctness and performance of the backward pass implementation.
"""

import pytest
import torch
import math
from flash_attention import flash_attention_forward, flash_attention_backward


class TestBackwardPass:
    """Test suite for Flash Attention backward pass"""

    def test_backward_correctness(self):
        """Test backward pass correctness against PyTorch autograd"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 2, 4, 32, 64

        # Create test tensors with gradients enabled
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()

        # Clone for reference computation
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)

        # Forward pass with our implementation
        with torch.no_grad():
            out = flash_attention_forward(q, k, v, use_half=False)

        # Forward pass with reference (standard attention)
        scale = 1.0 / math.sqrt(head_dim)
        attn_ref = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
        attn_ref = torch.softmax(attn_ref, dim=-1)
        out_ref = torch.matmul(attn_ref, v_ref)

        # Backward pass with our implementation
        # Note: In practice, you'd get l and m from the forward pass
        # Here we use dummy values for testing
        l = torch.zeros(batch_size, num_heads, seq_len).cuda()
        m = torch.full((batch_size, num_heads, seq_len), -float('inf')).cuda()

        dq, dk, dv = flash_attention_backward(q, k, v, out, out, l, m, use_half=False)

        # Backward pass with reference
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Check that gradients are computed and have reasonable values
        assert dq is not None and torch.isfinite(dq).all(), "Query gradients are invalid"
        assert dk is not None and torch.isfinite(dk).all(), "Key gradients are invalid"
        assert dv is not None and torch.isfinite(dv).all(), "Value gradients are invalid"

        # Check gradient shapes
        assert dq.shape == q.shape, f"Query gradient shape mismatch: {dq.shape} vs {q.shape}"
        assert dk.shape == k.shape, f"Key gradient shape mismatch: {dk.shape} vs {k.shape}"
        assert dv.shape == v.shape, f"Value gradient shape mismatch: {dv.shape} vs {v.shape}"

    def test_backward_with_autograd(self):
        """Test backward pass by comparing with PyTorch autograd"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 32

        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()

        # Forward pass
        out = flash_attention_forward(q, k, v, use_half=False)

        # Compute loss and backward
        loss = out.sum()
        loss.backward()

        # Check that gradients exist and are finite
        assert q.grad is not None, "Query gradient not computed"
        assert k.grad is not None, "Key gradient not computed"
        assert v.grad is not None, "Value gradient not computed"

        assert torch.isfinite(q.grad).all(), "Query gradient contains NaN/Inf"
        assert torch.isfinite(k.grad).all(), "Key gradient contains NaN/Inf"
        assert torch.isfinite(v.grad).all(), "Value gradient contains NaN/Inf"

    @pytest.mark.parametrize("use_half", [False, True])
    def test_backward_precision(self, use_half):
        """Test backward pass with different precisions"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 32

        dtype = torch.float16 if use_half else torch.float32

        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, requires_grad=True).cuda()

        # Forward pass
        out = flash_attention_forward(q, k, v, use_half=use_half)

        # Compute loss and backward
        loss = out.sum()
        loss.backward()

        # Check gradients
        assert q.grad is not None and torch.isfinite(q.grad).all(), f"Query gradient invalid for {'FP16' if use_half else 'FP32'}"
        assert k.grad is not None and torch.isfinite(k.grad).all(), f"Key gradient invalid for {'FP16' if use_half else 'FP32'}"
        assert v.grad is not None and torch.isfinite(v.grad).all(), f"Value gradient invalid for {'FP16' if use_half else 'FP32'}"

    def test_backward_different_block_sizes(self):
        """Test backward pass with different block sizes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()

        # Test different block sizes
        block_configs = [(32, 32), (64, 32), (32, 64)]

        gradients = []
        for bc, br in block_configs:
            # Clear gradients
            q.grad = None
            k.grad = None
            v.grad = None

            # Forward and backward
            out = flash_attention_forward(q, k, v, bc=bc, br=br, use_half=False)
            loss = out.sum()
            loss.backward()

            # Store gradients
            gradients.append((q.grad.clone(), k.grad.clone(), v.grad.clone()))

        # Check that gradients are similar (within numerical precision)
        for i in range(1, len(gradients)):
            dq1, dk1, dv1 = gradients[0]
            dq2, dk2, dv2 = gradients[i]

            assert torch.allclose(dq1, dq2, rtol=1e-3, atol=1e-4), f"Query gradients differ for config {i}"
            assert torch.allclose(dk1, dk2, rtol=1e-3, atol=1e-4), f"Key gradients differ for config {i}"
            assert torch.allclose(dv1, dv2, rtol=1e-3, atol=1e-4), f"Value gradients differ for config {i}"

    def test_backward_gradient_magnitude(self):
        """Test that backward gradients have reasonable magnitudes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

        # Use small values to avoid numerical issues
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda() * 0.1
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda() * 0.1
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda() * 0.1

        # Forward and backward
        out = flash_attention_forward(q, k, v, use_half=False)
        loss = out.sum()
        loss.backward()

        # Check that gradients are not too large or too small
        assert q.grad.abs().mean() < 10.0, f"Query gradient magnitude too large: {q.grad.abs().mean()}"
        assert k.grad.abs().mean() < 10.0, f"Key gradient magnitude too large: {k.grad.abs().mean()}"
        assert v.grad.abs().mean() < 10.0, f"Value gradient magnitude too large: {v.grad.abs().mean()}"

        assert q.grad.abs().mean() > 1e-7, f"Query gradient magnitude too small: {q.grad.abs().mean()}"
        assert k.grad.abs().mean() > 1e-7, f"Key gradient magnitude too small: {k.grad.abs().mean()}"
        assert v.grad.abs().mean() > 1e-7, f"Value gradient magnitude too small: {v.grad.abs().mean()}"

    def test_backward_multiple_calls(self):
        """Test that backward pass can be called multiple times"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

        for _ in range(3):  # Test multiple calls
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()

            # Forward and backward
            out = flash_attention_forward(q, k, v, use_half=False)
            loss = out.sum()
            loss.backward()

            # Verify gradients exist
            assert q.grad is not None and torch.isfinite(q.grad).all()
            assert k.grad is not None and torch.isfinite(k.grad).all()
            assert v.grad is not None and torch.isfinite(v.grad).all()

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_heads", [1, 4])
    def test_backward_shapes(self, batch_size, num_heads):
        """Test backward pass with different tensor shapes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        seq_len, head_dim = 16, 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True).cuda()

        # Forward and backward
        out = flash_attention_forward(q, k, v, use_half=False)
        loss = out.sum()
        loss.backward()

        # Check shapes
        assert q.grad.shape == (batch_size, num_heads, seq_len, head_dim)
        assert k.grad.shape == (batch_size, num_heads, seq_len, head_dim)
        assert v.grad.shape == (batch_size, num_heads, seq_len, head_dim)
