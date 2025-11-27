# API Reference

## Core Functions

### `flash_attention_forward(Q, K, V, Bc=32, Br=32, use_half=False)`

The main forward pass function for Flash Attention.

**Parameters:**
- `Q` (torch.Tensor): Query tensor of shape `[B, H, N, D]`
- `K` (torch.Tensor): Key tensor of shape `[B, H, N, D]`
- `V` (torch.Tensor): Value tensor of shape `[B, H, N, D]`
- `Bc` (int, optional): Block size for columns (default: 32)
- `Br` (int, optional): Block size for rows (default: 32)
- `use_half` (bool, optional): Use FP16 precision (default: False)

**Returns:**
- `torch.Tensor`: Output tensor of shape `[B, H, N, D]`

**Example:**
```python
import torch
from flash_attention import flash_attention_forward

Q = torch.randn(8, 16, 1024, 64).cuda()
K = torch.randn(8, 16, 1024, 64).cuda()
V = torch.randn(8, 16, 1024, 64).cuda()

output = flash_attention_forward(Q, K, V, Bc=64, Br=32, use_half=False)
```

---

### `flash_attention_backward(Q, K, V, O, dO, l, m, Bc=32, Br=32, use_half=False)`

Backward pass function for computing gradients.

**Parameters:**
- `Q` (torch.Tensor): Query tensor `[B, H, N, D]`
- `K` (torch.Tensor): Key tensor `[B, H, N, D]`
- `V` (torch.Tensor): Value tensor `[B, H, N, D]`
- `O` (torch.Tensor): Output from forward pass `[B, H, N, D]`
- `dO` (torch.Tensor): Gradient w.r.t. output `[B, H, N, D]`
- `l` (torch.Tensor): Row sums from forward pass `[B, H, N]`
- `m` (torch.Tensor): Row maxima from forward pass `[B, H, N]`
- `Bc` (int, optional): Block size for columns (default: 32)
- `Br` (int, optional): Block size for rows (default: 32)
- `use_half` (bool, optional): Use FP16 precision (default: False)

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: Gradients `(dQ, dK, dV)`

---

## Class Interface

### `FlashAttention(bc=32, br=32, use_half=False)`

PyTorch module interface for Flash Attention.

**Parameters:**
- `bc` (int, optional): Block size for columns (default: 32)
- `br` (int, optional): Block size for rows (default: 32)
- `use_half` (bool, optional): Use FP16 precision (default: False)

**Methods:**
- `forward(Q, K, V)`: Forward pass
- `backward(Q, K, V, O, dO, l, m)`: Backward pass

**Example:**
```python
from flash_attention import FlashAttention

# Create module
flash_attn = FlashAttention(bc=64, br=32, use_half=True)

# Use in forward pass
output = flash_attn(Q, K, V)
```

---

## Utility Functions

### `get_optimal_block_sizes(sequence_length, head_dim)`

Get optimal block sizes for given sequence length and head dimension.

**Parameters:**
- `sequence_length` (int): Length of the sequence
- `head_dim` (int): Dimension of each attention head

**Returns:**
- `Tuple[int, int]`: Optimal `(Bc, Br)` block sizes

**Example:**
```python
from flash_attention import get_optimal_block_sizes

bc, br = get_optimal_block_sizes(1024, 128)
print(f"Optimal block sizes: Bc={bc}, Br={br}")
```

---

### `estimate_memory_usage(batch_size, num_heads, seq_len, head_dim, use_half=False)`

Estimate memory usage for Flash Attention computation.

**Parameters:**
- `batch_size` (int): Batch size
- `num_heads` (int): Number of attention heads
- `seq_len` (int): Sequence length
- `head_dim` (int): Head dimension
- `use_half` (bool, optional): Use FP16 precision (default: False)

**Returns:**
- `dict`: Memory usage information with keys:
  - `'input_memory'`: Memory for input tensors (bytes)
  - `'internal_memory'`: Memory for internal computations (bytes)
  - `'output_memory'`: Memory for output tensor (bytes)
  - `'total_memory'`: Total memory required (bytes)
  - `'total_memory_mb'`: Total memory in MB

**Example:**
```python
from flash_attention import estimate_memory_usage

mem_info = estimate_memory_usage(8, 16, 1024, 64, use_half=True)
print(f"Total memory: {mem_info['total_memory_mb']:.1f} MB")
```

---

## Tensor Shapes and Data Types

### Input/Output Shapes

All tensors follow the standard attention convention:
- **Batch dimension (B)**: Number of sequences in a batch
- **Head dimension (H)**: Number of attention heads
- **Sequence dimension (N)**: Length of the sequence
- **Feature dimension (D)**: Dimension of each head

### Supported Data Types

- **torch.float32**: Standard precision (FP32)
- **torch.float16**: Half precision (FP16)

### Device Support

- **CUDA**: Primary target with optimized kernels
- **CPU**: Fallback implementation for testing

---

## Performance Tuning

### Block Size Selection

Block sizes (`Bc`, `Br`) control the tiling strategy:
- **Smaller blocks**: Better memory efficiency, may be slower
- **Larger blocks**: Potentially faster but use more shared memory
- **Use `get_optimal_block_sizes()`** for automatic tuning

### Precision Selection

- **FP32**: Higher accuracy, uses more memory and compute
- **FP16**: Faster, lower memory usage, slightly less accurate

### Memory Considerations

- Flash Attention uses **O(N × D)** memory instead of **O(N²)**
- FP16 reduces memory usage by ~50%
- Block sizes affect shared memory requirements

---

## Error Handling

The implementation includes comprehensive error checking:

- **CUDA errors**: Automatically checked with detailed error messages
- **Memory limits**: Validates shared memory requirements
- **Tensor shapes**: Ensures input tensors have compatible dimensions
- **Data types**: Validates supported precision modes

---

## Examples

### Basic Usage
```python
import torch
from flash_attention import flash_attention_forward

# Create input tensors
Q = torch.randn(4, 8, 512, 64).cuda()
K = torch.randn(4, 8, 512, 64).cuda()
V = torch.randn(4, 8, 512, 64).cuda()

# Forward pass
output = flash_attention_forward(Q, K, V)
```

### Training Loop
```python
import torch.nn as nn
from flash_attention import FlashAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = FlashAttention(use_half=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x
```

### Memory Optimization
```python
from flash_attention import estimate_memory_usage, get_optimal_block_sizes

# Estimate memory for your model
batch_size, num_heads, seq_len, head_dim = 8, 16, 2048, 64
mem_info = estimate_memory_usage(batch_size, num_heads, seq_len, head_dim, use_half=True)

if mem_info['total_memory_mb'] > 1000:  # More than 1GB
    print("Consider using smaller batch size or FP16")

# Get optimal block sizes
bc, br = get_optimal_block_sizes(seq_len, head_dim)
