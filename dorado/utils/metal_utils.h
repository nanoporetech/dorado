#pragma once
#include <Metal/Metal.hpp>
#include <torch/torch.h>

using namespace MTL;

Buffer *create_buffer(MTL::Device *device, size_t length);

template <typename T>
MTL::Buffer *create_buffer(MTL::Device *device, T *ptr, size_t length) {
    return device->newBuffer(ptr, length * sizeof(T), MTL::ResourceStorageModeShared);
}

ComputePipelineState *make_cps(MTL::Device *device, std::string name);
void launch_kernel(ComputePipelineState *cps,
                   CommandQueue *cq,
                   std::vector<Buffer *> buffers,
                   long threadgroups,
                   long threads_per_threadroup);
void launch_kernel_no_wait(ComputePipelineState *cps,
                           CommandBuffer *cb,
                           std::vector<Buffer *> buffers,
                           long threadgroups,
                           long threads_per_threadgroup);

void lock_mtl_device();
void unlock_mtl_device();

MTL::Device *get_mtl_device();
MTL::Buffer *mtl_for_tensor(const torch::Tensor &t);
MTL::Buffer *extract_mtl_from_tensor(torch::Tensor &t);
