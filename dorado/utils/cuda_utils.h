#pragma once

#include "../nn/CRFModel.h"

#include <nvtx3/nvtx3.hpp>
#include <torch/torch.h>

#include <array>
#include <mutex>
#include <optional>
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
// each device (e.g ["cuda:0", "cuda:2", ..., "cuda:7"]
std::vector<std::string> parse_cuda_device_string(std::string device_string);

// Reports the amount of available memory (in bytes) for a given device.
size_t available_memory(torch::Device device);

// Print `label` and stats for Torch CUDACachingAllocator to stderr. Useful for tracking down
// where Torch allocates GPU memory.
void print_cuda_alloc_info(const std::string &label);

int auto_gpu_batch_size(torch::nn::ModuleHolder<torch::nn::AnyModule> module,
                        const dorado::CRFModelConfig &model_config,
                        const torch::TensorOptions &options,
                        int batch_size_granularity,
                        float memory_limit_fraction);

void matmul_f16(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C);

// Deal with a result from a cudaGetLastError call.  May raise an exception to provide information to the user.
void handle_cuda_result(int cuda_result);

#if CUDA_PROFILE_TO_CERR
// Times range and prints it to console so you don't have to generate a QDREP file to perform
// basic profiling
class ScopedProfileRange {
public:
    explicit ScopedProfileRange(const char *label) : m_nvtx_range(label), m_label(label) {
        m_stream = at::cuda::getCurrentCUDAStream().stream();
        handle_cuda_result(cudaEventCreate(&m_start));
        handle_cuda_result(cudaEventRecord(m_start, m_stream));
        m_active = true;
    }

    ~ScopedProfileRange() { finish(); }

private:
    void finish() {
        if (!m_active) {
            return;
        }
        cudaEvent_t stop;
        handle_cuda_result(cudaEventCreate(&stop));
        handle_cuda_result(cudaEventRecord(stop, m_stream));
        handle_cuda_result(cudaEventSynchronize(stop));
        float timeMs = 0.0f;
        handle_cuda_result(cudaEventElapsedTime(&timeMs, m_start, stop));
        handle_cuda_result(cudaEventDestroy(m_start));
        handle_cuda_result(cudaEventDestroy(stop));
        std::cerr << "[" << m_label << " " << timeMs << " ms]" << std::endl;
        m_active = false;
    }

    nvtx3::scoped_range m_nvtx_range;
    const char *m_label;
    cudaStream_t m_stream;
    cudaEvent_t m_start;
    bool m_active;
};
#else  // if CUDA_PROFILE_TO_CERR
using ScopedProfileRange = nvtx3::scoped_range;
#endif

namespace details {
// Exposed in the header for testability
std::optional<std::array<int, 3>> try_select_max_batch_sizes(
        std::vector<int> const &breakpoints,
        std::vector<std::array<int, 3>> const &batch_sizes,
        int available_memory_gb);

void matmul_f16_cublas(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C);
void matmul_f16_torch(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C);

}  //  namespace details

}  // namespace dorado::utils
