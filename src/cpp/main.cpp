#include <torch/extension.h>
#include <vector>

// Forward and backward function declarations
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                    int Bc, int Br, bool use_half);
std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor dO, torch::Tensor l, torch::Tensor m,
    int Bc, int Br, bool use_half);

// Legacy function for backward compatibility
torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    return flash_attention_forward(q, k, v, 32, 32, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward (legacy)");
    m.def("flash_attention_forward", torch::wrap_pybind_function(flash_attention_forward),
          "Modern flash attention forward pass with configurable parameters",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("Bc") = 32, py::arg("Br") = 32, py::arg("use_half") = false);
    m.def("flash_attention_backward", torch::wrap_pybind_function(flash_attention_backward),
          "Flash attention backward pass for training",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("dO"),
          py::arg("l"), py::arg("m"),
          py::arg("Bc") = 32, py::arg("Br") = 32, py::arg("use_half") = false);
}
