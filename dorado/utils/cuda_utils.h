#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <mutex>
#include <string>
#include <vector>

namespace dorado::utils {

// Returns a lock providing exclusive access to the GPU with the specified index.
// In cases where > 1 model is being used, this can prevent more than one from
// attempting to allocate GPU memory on, or submit work to, the device in question.
// Once the returned lock goes out of scope or is explicitly unlocked,
// the GPU is available to other users again.
std::unique_lock<std::mutex> acquire_gpu_lock(int gpu_index, bool use_lock);

// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a vector of strings, one for
// each device (e.g ["cuda:0", "cuda:2", ..., "cuda:7"]. This function will validate that the device IDs
// exist and will raise an exception if there is any issue with the string.
std::vector<std::string> parse_cuda_device_string(std::string device_string);

struct CUDADeviceInfo {
    size_t free_mem, total_mem;
    int device_id;
    int compute_cap_major, compute_cap_minor;
    cudaDeviceProp device_properties;
    bool in_use;
};

// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a vector of CUDADeviceInfo for all
// visible devices on the host machine, with information on whether they are in use or not.
std::vector<CUDADeviceInfo> get_cuda_device_info(std::string device_string);

// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a string containing
// the set of types of gpu that will be used.
std::string get_cuda_gpu_names(std::string device_string);

// Reports the amount of available memory (in bytes) for a given device.
size_t available_memory(torch::Device device);

// Print `label` and stats for Torch CUDACachingAllocator to stderr. Useful for tracking down
// where Torch allocates GPU memory.
void print_cuda_alloc_info(const std::string &label);

void matmul_f16(const at::Tensor &A, const at::Tensor &B, at::Tensor &C);

// Deal with a result from a cudaGetLastError call.  May raise an exception to provide information to the user.
void handle_cuda_result(int cuda_result);

namespace details {
void matmul_f16_cublas(const at::Tensor &A, const at::Tensor &B, at::Tensor &C);
void matmul_f16_torch(const at::Tensor &A, const at::Tensor &B, at::Tensor &C);

}  //  namespace details

}  // namespace dorado::utils
