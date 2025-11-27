import math
import time
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the modern CUDA kernel as a python module
print("Loading modern Flash Attention CUDA extension...")
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'],
                   extra_cuda_cflags=['-O2', '-arch=sm_80', '--use_fast_math'],
                   extra_cflags=['-std=c++17'])

print("CUDA extension loaded successfully!")

# Test configurations
configs = [
    {"batch_size": 16, "n_head": 12, "seq_len": 64, "head_embd": 64, "name": "Small"},
    {"batch_size": 8, "n_head": 16, "seq_len": 128, "head_embd": 128, "name": "Medium"},
    {"batch_size": 4, "n_head": 32, "seq_len": 256, "head_embd": 64, "name": "Large"},
]

def manual_attn(q, k, v):
    """Reference implementation using standard attention"""
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def benchmark_attention(config, use_half=False, Bc=32, Br=32):
    """Benchmark different attention implementations"""
    print(f"\n=== Testing {config['name']} Configuration (half={use_half}, Bc={Bc}, Br={Br}) ===")

    # Create tensors
    if use_half:
        q = torch.randn(config['batch_size'], config['n_head'], config['seq_len'], config['head_embd'],
                       dtype=torch.float16).cuda()
        k = torch.randn(config['batch_size'], config['n_head'], config['seq_len'], config['head_embd'],
                       dtype=torch.float16).cuda()
        v = torch.randn(config['batch_size'], config['n_head'], config['seq_len'], config['head_embd'],
                       dtype=torch.float16).cuda()
    else:
        q = torch.randn(config['batch_size'], config['n_head'], config['seq_len'], config['head_embd']).cuda()
        k = torch.randn(config['batch_size'], config['n_head'], config['seq_len'], config['head_embd']).cuda()
        v = torch.randn(config['batch_size'], config['n_head'], config['seq_len'], config['head_embd']).cuda()

    print(f"Tensor shapes: Q={q.shape}, K={q.shape}, V={v.shape}")
    print(f"Memory usage: {q.numel() * q.element_size() / 1024 / 1024:.1f} MB per tensor")

    # Manual attention benchmark
    print("\n--- Manual Attention (Reference) ---")
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        manual_result = manual_attn(q.float(), k.float(), v.float())
    torch.cuda.synchronize()
    manual_time = time.time() - start_time
    print(".4f")

    # Modern flash attention benchmark
    print("\n--- Modern Flash Attention ---")
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        flash_result = minimal_attn.flash_attention_forward(q, k, v, Bc, Br, use_half)
    torch.cuda.synchronize()
    flash_time = time.time() - start_time
    print(".4f")
    print(".2f")

    # Accuracy check
    if use_half:
        accuracy = torch.allclose(flash_result.float(), manual_result, rtol=0, atol=1e-02)
    else:
        accuracy = torch.allclose(flash_result, manual_result, rtol=0, atol=1e-02)
    print(f"Accuracy check: {'PASS' if accuracy else 'FAIL'}")

    return flash_time, manual_time

def test_backward_pass():
    """Test backward pass functionality"""
    print("\n=== Testing Backward Pass ===")

    batch_size, n_head, seq_len, head_embd = 4, 8, 32, 64
    q = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()

    # Forward pass
    out = minimal_attn.flash_attention_forward(q, k, v, 32, 32, False)

    # Backward pass
    torch.cuda.synchronize()
    start_time = time.time()
    out.sum().backward()
    torch.cuda.synchronize()
    backward_time = time.time() - start_time

    print(".4f")
    print(f"Gradients computed: Q={q.grad is not None}, K={k.grad is not None}, V={v.grad is not None}")

    return backward_time

# Run benchmarks
print("🚀 Modern Flash Attention Benchmark Suite")
print("=" * 50)

for config in configs:
    # Test different configurations
    for use_half in [False, True]:
        for Bc, Br in [(32, 32), (64, 32), (32, 64)]:
            try:
                flash_time, manual_time = benchmark_attention(config, use_half, Bc, Br)
            except Exception as e:
                print(f"❌ Error testing {config['name']} (half={use_half}): {e}")

# Test backward pass
try:
    backward_time = test_backward_pass()
except Exception as e:
    print(f"❌ Error testing backward pass: {e}")

print("\n🎉 Benchmark suite completed!")
print("\nKey improvements in this version:")
print("✅ Modern CUDA features (cooperative groups, async memcpy)")
print("✅ Backward pass support for training")
print("✅ Half precision (FP16) support")
print("✅ Dynamic block sizes")
print("✅ Better memory management and error handling")
print("✅ C++17 features and cleaner code structure")
