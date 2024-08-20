#pragma once

// Must come before Metal.hpp or we get warnings about __attribute__
#include <ostream>

// Some NS types make use of tagged pointers which aren't aligned and trip up UBSan.
#pragma clang attribute push(__attribute__((no_sanitize("alignment"))), apply_to = function)
#include <Metal/Metal.hpp>
#pragma clang attribute pop

#include <ATen/core/TensorBody.h>

#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace dorado::utils {

template <typename MetalObject>
void name_mtl_object(MetalObject &&obj, const char *name) {
    obj->setLabel(NS::String::string(name, NS::ASCIIStringEncoding));
}

// Returns an uninitialised MTL::Buffer of length bytes.
NS::SharedPtr<MTL::Buffer> create_buffer(MTL::Device *device, size_t length);

// Returns a MTL::Buffer holding the content of the supplied std::vector.
template <typename T>
NS::SharedPtr<MTL::Buffer> create_vec_buffer(MTL::Device *const device, const std::vector<T> &vec) {
    return NS::TransferPtr(
            device->newBuffer(vec.data(), vec.size() * sizeof(T), MTL::ResourceStorageModeShared));
}

using MetalConstant = std::variant<int, bool, float>;

// Returns a ComputePipelineState object created from the named kernel and
// given constants.  If max_total_threads_per_tg != -1, the value overrides
// the default (and the shader should not itself specify the value).
NS::SharedPtr<MTL::ComputePipelineState> make_cps(
        MTL::Device *device,
        const std::string &name,
        const std::vector<std::tuple<std::string, MetalConstant>> &named_constants,
        const std::optional<int> max_total_threads_per_tg);

void launch_kernel(MTL::ComputePipelineState *cps,
                   MTL::CommandQueue *cq,
                   const std::vector<MTL::Buffer *> &buffers,
                   const std::vector<int> &tg_buffer_lens,
                   long threadgroups,
                   long threads_per_threadroup);

MTL::CommandBuffer *next_command_buffer(MTL::CommandQueue *queue, int try_count);

void launch_kernel_no_wait(MTL::ComputePipelineState *cps,
                           MTL::CommandBuffer *cb,
                           const std::vector<MTL::Buffer *> &buffers,
                           const std::vector<int> &tg_buffer_lens,
                           long threadgroups,
                           long threads_per_threadgroup);

// Returns true on success.
bool finishCommandBuffer(const char *label, MTL::CommandBuffer *cb, int try_count);

NS::SharedPtr<MTL::Device> get_mtl_device();
int get_mtl_device_core_count();
int get_apple_cpu_perf_core_count();
size_t get_apple_physical_memory_bytes();
MTL::Buffer *mtl_for_tensor(const at::Tensor &t);
NS::SharedPtr<MTL::Buffer> extract_mtl_from_tensor(at::Tensor &&t);

// On construction, creates an autorelease pool for the current thread.
// On destruction, drains the autorelease pool.
class ScopedAutoReleasePool {
public:
    ScopedAutoReleasePool();
    ~ScopedAutoReleasePool();

private:
    id m_autorelease_pool;
};

}  // namespace dorado::utils
