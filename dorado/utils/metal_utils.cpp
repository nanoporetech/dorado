#include "metal_utils.h"

#include <Metal/Metal.hpp>
#include <mach-o/dyld.h>
#include <sys/syslimits.h>
#include <torch/torch.h>

#include <filesystem>

using namespace std;
using namespace MTL;

namespace fs = std::filesystem;

NS::String *get_library_location() {
    char ns_path[PATH_MAX + 1];
    uint32_t size = sizeof(ns_path);
    _NSGetExecutablePath(ns_path, &size);

    fs::path exepth{ns_path};
    fs::path mtllib{"../lib/default.metallib"};
    fs::path fspath = exepth.parent_path() / mtllib;

    return NS::String::string(fspath.c_str(), NS::ASCIIStringEncoding);
}

MTL::Buffer *create_buffer(MTL::Device *device, size_t length) {
    return device->newBuffer(length, MTL::ResourceStorageModeShared);
}

ComputePipelineState *make_cps(Device *device, const std::string name) {
    NS::Error *error;
    auto default_library = device->newDefaultLibrary();

    if (!default_library) {
        auto lib_path = get_library_location();
        default_library = device->newLibrary(lib_path, &error);
        if (!default_library) {
            throw std::runtime_error("Failed to load metallib library.");
        }
    }

    auto kernel_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
    auto kernel = default_library->newFunction(kernel_name);

    if (!kernel) {
        throw std::runtime_error("Failed to find the kernel: " + name);
    }
    auto cps = device->newComputePipelineState(kernel, &error);

    if (cps == NULL) {
        auto e_code = to_string(((int)error->code()));
        auto e_str = error->domain()->cString(NS::ASCIIStringEncoding);
        throw std::runtime_error("failed to build compute pipeline for " + name + " - " + e_str +
                                 ": error " + e_code);
    }

    return cps;
}

void launch_kernel(ComputePipelineState *pipeline,
                   CommandQueue *command_queue,
                   vector<Buffer *> buffers,
                   long threadgroups,
                   long threads_per_threadgroup) {
    auto command_buffer = command_queue->commandBuffer();
    launch_kernel_no_wait(pipeline, command_buffer, buffers, threadgroups, threads_per_threadgroup);

    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}

void launch_kernel_no_wait(ComputePipelineState *pipeline,
                           CommandBuffer *command_buffer,
                           vector<Buffer *> buffers,
                           long threadgroups,
                           long threads_per_threadgroup) {
    auto compute_encoder = command_buffer->computeCommandEncoder();
    compute_encoder->setComputePipelineState(pipeline);

    for (auto i = 0; i < (int)buffers.size(); i++) {
        compute_encoder->setBuffer(buffers[i], 0, i);
    }

    compute_encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1),
                                          MTL::Size(threads_per_threadgroup, 1, 1));
    compute_encoder->memoryBarrier(BarrierScopeBuffers);
    compute_encoder->endEncoding();
}

static MTL::Device *mtl_device{nullptr};

struct MTLAllocator : torch::Allocator {
    virtual ~MTLAllocator() = default;

    virtual torch::DataPtr allocate(size_t n) const {
        if (n == 0) {
            return torch::DataPtr(nullptr, torch::DeviceType::CPU);
        } else if (n >= (size_t(1) << 32)) {
            return torch::DataPtr(new char[n], torch::DeviceType::CPU);
        }
        auto buffer = mtl_device->newBuffer(n, MTL::ResourceStorageModeShared);
        return torch::DataPtr(buffer->contents(), buffer, &deleter, torch::DeviceType::CPU);
    }

    static void deleter(void *ptr) { ((MTL::Buffer *)ptr)->release(); }
};
static MTLAllocator mtl_allocator;

static std::mutex mtl_device_mutex;
static thread_local std::unique_ptr<std::unique_lock<std::mutex>> mtl_device_lock;

void lock_mtl_device() {
    if (!mtl_device_lock) {
        mtl_device_lock = std::make_unique<std::unique_lock<std::mutex>>(mtl_device_mutex);
    } else {
        mtl_device_lock->lock();
    }
}

void unlock_mtl_device() { mtl_device_lock->unlock(); }

MTL::Device *get_mtl_device() {
    if (mtl_device == nullptr) {
        mtl_device = MTL::CreateSystemDefaultDevice();
        torch::SetAllocator(torch::DeviceType::CPU, &mtl_allocator);
    }
    return mtl_device;
}

MTL::Buffer *mtl_for_tensor(const torch::Tensor &x) {
    auto ptr = (MTL::Buffer *)(x.storage().data_ptr().get_context());
    assert(ptr != nullptr);
    return ptr;
}

MTL::Buffer *extract_mtl_from_tensor(torch::Tensor &x) {
    auto bfr = mtl_for_tensor(x);
    bfr->retain();
    x.reset();
    return bfr;
}
