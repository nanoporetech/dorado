#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

namespace dorado::utils {

// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a vector of strings, one for
// each device (e.g ["cuda:0", "cuda:2", ..., "cuda:7"]
std::vector<std::string> parse_cuda_device_string(std::string device_string);

// Reports the amount of available memory (in bytes) for a given device.
size_t available_memory(std::string device);
// Given a vector of devices in the format as returned by parse_cuda_device_string this function
// will report the amount of available memory (in bytes) for all devices.
std::vector<size_t> available_memory(std::vector<std::string> devices);
int auto_gpu_batch_size(std::string model_path, std::vector<std::string> devices);

void cublas_matmul_f16(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C);

}  // namespace dorado::utils
