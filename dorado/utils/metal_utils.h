#pragma once
#include <Metal/Metal.hpp>
#include <torch/torch.h>

#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace dorado::utils {

// Returns an uninitialised MTL::Buffer of length bytes.
MTL::Buffer *create_buffer(MTL::Device *device, size_t length);

// Returns a MTL::Buffer holding the content of the supplied std::vector.
template <typename T>
MTL::Buffer *create_vec_buffer(MTL::Device *const device, const std::vector<T> &vec) {
    return device->newBuffer(vec.data(), vec.size() * sizeof(T), MTL::ResourceStorageModeShared);
}

using MetalConstant = std::variant<int, bool, float>;

// Returns a ComputePipelineState object created from the named kernel and
// given constants.  If max_total_threads_per_tg != -1, the value overrides
// the default (and the shader should not itself specify the value).
MTL::ComputePipelineState *make_cps(
        MTL::Device *device,
        const std::string &name,
        const std::vector<std::tuple<std::string, MetalConstant>> &named_constants,
        const int max_total_threads_per_tg = -1);

void launch_kernel(MTL::ComputePipelineState *cps,
                   MTL::CommandQueue *cq,
                   const std::vector<MTL::Buffer *> &buffers,
                   const std::vector<int> &tg_buffer_lens,
                   long threadgroups,
                   long threads_per_threadroup);

void launch_kernel_no_wait(MTL::ComputePipelineState *cps,
                           MTL::CommandBuffer *cb,
                           const std::vector<MTL::Buffer *> &buffers,
                           const std::vector<int> &tg_buffer_lens,
                           long threadgroups,
                           long threads_per_threadgroup);

MTL::Device *get_mtl_device();
int get_mtl_device_core_count();
int get_apple_cpu_perf_core_count();
MTL::Buffer *mtl_for_tensor(const torch::Tensor &t);
MTL::Buffer *extract_mtl_from_tensor(torch::Tensor &t);
int auto_gpu_batch_size();

}  // namespace dorado::utils
