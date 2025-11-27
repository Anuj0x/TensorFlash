#pragma once

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <type_traits>

// Configuration structure for kernel parameters
struct FlashAttentionConfig {
    int Bc, Br;  // Block sizes for columns and rows
    bool use_half;  // Use half precision
    bool use_tensor_cores;  // Use tensor core acceleration
};

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Half precision utilities
__device__ __forceinline__ half2 __hadd2_half(half a, half b) {
    return __hadd2(__half2half2(a), __half2half2(b));
}

__device__ __forceinline__ float __half2float(half2 a) {
    return __half2float(__low2half(a)) + __half2float(__high2half(a));
}

// Forward pass kernel declaration
template<typename T>
__global__ void flash_attention_forward_kernel(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
    const int N, const int d, const int Tc, const int Tr,
    const int Bc, const int Br, const float softmax_scale,
    float* __restrict__ l, float* __restrict__ m, T* __restrict__ O,
    const FlashAttentionConfig config);

// Backward pass kernel declaration
template<typename T>
__global__ void flash_attention_backward_kernel(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
    const T* __restrict__ O, const float* __restrict__ dO,
    const float* __restrict__ l, const float* __restrict__ m,
    const int N, const int d, const int Tc, const int Tr,
    const int Bc, const int Br, const float softmax_scale,
    T* __restrict__ dQ, T* __restrict__ dK, T* __restrict__ dV,
    const FlashAttentionConfig config);

// Host function declarations
template<typename T>
torch::Tensor flash_attention_forward_impl(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                         const FlashAttentionConfig& config);

template<typename T>
std::vector<torch::Tensor> flash_attention_backward_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor dO, torch::Tensor l, torch::Tensor m,
    const FlashAttentionConfig& config);

// Public interface functions
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                    int Bc = 32, int Br = 32, bool use_half = false);

std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor dO, torch::Tensor l, torch::Tensor m,
    int Bc = 32, int Br = 32, bool use_half = false);
