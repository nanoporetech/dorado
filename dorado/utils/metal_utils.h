#pragma once
#include <Metal/Metal.hpp>
#include <torch/torch.h>

#include <string>
#include <vector>

// Returns an uninitialised MTL::Buffer of length bytes.
MTL::Buffer *create_buffer(MTL::Device *device, size_t length);

// Returns a MTL::Buffer holding the content of the supplied std::vector.
template <typename T>
MTL::Buffer *create_vec_buffer(MTL::Device *const device, const std::vector<T> &vec) {
    return device->newBuffer(vec.data(), vec.size() * sizeof(T), MTL::ResourceStorageModeShared);
}

MTL::ComputePipelineState *make_cps(MTL::Device *device, std::string name);
void launch_kernel(MTL::ComputePipelineState *cps,
                   MTL::CommandQueue *cq,
                   std::vector<MTL::Buffer *> buffers,
                   long threadgroups,
                   long threads_per_threadroup);
void launch_kernel_no_wait(MTL::ComputePipelineState *cps,
                           MTL::CommandBuffer *cb,
                           std::vector<MTL::Buffer *> buffers,
                           long threadgroups,
                           long threads_per_threadgroup);

MTL::Device *get_mtl_device();
int get_mtl_device_core_count();
int get_apple_cpu_perf_core_count();
MTL::Buffer *mtl_for_tensor(const torch::Tensor &t);
MTL::Buffer *extract_mtl_from_tensor(torch::Tensor &t);
