#pragma once

// Must come before Metal.hpp or we get warnings about __attribute__
#include <ostream>

// Some NS types make use of tagged pointers which aren't aligned and trip up UBSan.
#pragma clang attribute push(__attribute__((no_sanitize("alignment"))), apply_to = function)
#include <Metal/Metal.hpp>
#pragma clang attribute pop

#include "utils/PostCondition.h"

#include <ATen/core/TensorBody.h>
#include <os/signpost.h>

#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

// Helper to capture "points" (ranges) of interest to help visualise bottlenecks.
// Usage:
//   // Create a global ID for what you want to track.
//   CREATE_POINT_OF_INTEREST_ID(my_id);
//   void function(int i) {
//     // Use the ID inside a scope to measure how long it takes.
//     POINT_OF_INTEREST_SCOPE(my_id, whole_scope);
//     do_thing();
//     {
//       // You can provide printf-style args to a scope too to help diagnose issues.
//       POINT_OF_INTEREST_SCOPE(my_id, whole_scope, "calling expensive_function(%i)", i);
//       expensive_function(i);
//     }
//   }
// Not technically metal, but only usable with Instruments.
#define CREATE_POINT_OF_INTEREST_ID(id)                                                        \
    static os_log_t _s_##id##_os_log = os_log_create(#id, OS_LOG_CATEGORY_POINTS_OF_INTEREST); \
    static_assert(true, "Force semicolon")
#define POINT_OF_INTEREST_SCOPE(id, name, ...)                                           \
    os_signpost_id_t _poi_id_##name = os_signpost_id_generate(_s_##id##_os_log);         \
    os_signpost_interval_begin(_s_##id##_os_log, _poi_id_##name, #name, "" __VA_ARGS__); \
    auto _poi_scope_##name = dorado::utils::PostCondition(                               \
            [&] { os_signpost_interval_end(_s_##id##_os_log, _poi_id_##name, #name); }); \
    static_assert(true, "Force semicolon")

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
bool run_command_buffer(const char *label, MTL::CommandBuffer *cb, int try_count);

NS::SharedPtr<MTL::Device> get_mtl_device();
int get_mtl_device_core_count();
int get_apple_cpu_perf_core_count();
size_t get_apple_physical_memory_bytes();
MTL::Buffer *mtl_for_tensor(const at::Tensor &t);
NS::SharedPtr<MTL::Buffer> extract_mtl_from_tensor(at::Tensor &&t);

// On construction, creates an autorelease pool for the current thread.
// On destruction, drains the autorelease pool.
class ScopedAutoReleasePool {
    id m_autorelease_pool;

    ScopedAutoReleasePool(const ScopedAutoReleasePool &) = delete;
    ScopedAutoReleasePool &operator=(const ScopedAutoReleasePool &) = delete;

public:
    ScopedAutoReleasePool();
    ~ScopedAutoReleasePool();
};

// Capture work on a device between 2 points.
// A path can be provided to dump the capture to a file.
class ScopedMetalCapture {
    MTL::Device *m_device = nullptr;

    ScopedMetalCapture(const ScopedMetalCapture &) = delete;
    ScopedMetalCapture &operator=(const ScopedMetalCapture &) = delete;

public:
    ScopedMetalCapture(MTL::Device *device, const std::optional<std::filesystem::path> &path);
    ~ScopedMetalCapture();

    // Traces only show the values in the buffers before the kernel executes,
    // so use this to insert a marker after a call to inspect values.
    void inspect_buffer(MTL::Buffer *buffer, MTL::CommandBuffer *cb);
};

}  // namespace dorado::utils
