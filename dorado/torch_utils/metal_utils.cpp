#include "metal_utils.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach-o/dyld.h>
#include <objc/objc-runtime.h>
#include <spdlog/spdlog.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>
#include <torch/version.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

using namespace MTL;

namespace fs = std::filesystem;

namespace {

// Allows less ugliness in use of std::visit.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Note: NS::String objects created via NS::String::string are placed in the autorelease pool,
// which means they will be released at a later time dictated by the autorelease pool setup.
// Setting up an NS::SharedPtr via NS::TransferPtr to hold them will result in an invalid attempt
// to free them a second time, entailing sending a message to a destroyed object, generally
// leading to invalid address accesses.  This is in contrast to other NSObjects here created
// using methods beginning with Create, alloc, new, which do require releasing via NS::SharedPtr
// or other means.
// Functions here that create autorelease objects should be called with an autorelease pool set up,
// which on MacOS isn't the case unless something like ScopedAutoReleasePool is used.

CREATE_POINT_OF_INTEREST_ID(metal_utils);

void report_error(const NS::Error *error, const char *function) {
    if (error == nil) {
        return;
    }
    auto safe_c_str = [](NS::String *str) {
        const char *c_str = str ? str->cString(NS::ASCIIStringEncoding) : nullptr;
        return c_str ? c_str : "<none>";
    };
    spdlog::error("function={}, code={}, domain={}, description={}", function, error->code(),
                  safe_c_str(error->domain()), safe_c_str(error->localizedDescription()));
}

#define wrap_func_with_err(func_with_err, ...)             \
    ({                                                     \
        NS::Error *error_ = nil;                           \
        auto result = func_with_err(__VA_ARGS__, &error_); \
        report_error(error_, #func_with_err);              \
        result;                                            \
    })

auto load_kernels(MTL::Device *const device) {
    char ns_path[PATH_MAX + 1];
    uint32_t size = PATH_MAX;
    if (_NSGetExecutablePath(ns_path, &size) < 0) {
        throw std::runtime_error("Failed to build path to kernels");
    }
    ns_path[size] = '\0';
    fs::path exepth{ns_path};

    // Check the default (ie compiled into the app)
    auto kernels = NS::TransferPtr(device->newDefaultLibrary());
    if (kernels) {
        spdlog::trace("Using default metal library");
        return kernels;
    }

    // Check in the lib folder.
    auto fspath = exepth.parent_path() / "../lib/default.metallib";
    auto lib_path = NS::String::string(fspath.c_str(), NS::ASCIIStringEncoding);
    kernels = NS::TransferPtr(wrap_func_with_err(device->newLibrary, lib_path));
    if (kernels) {
        spdlog::trace("Using metal library at {}", fspath.string());
        return kernels;
    }

    throw std::runtime_error("Failed to load metallib library");
}

// Retrieves a single int64_t property associated with the given class/name.
// Returns empty std::optional on failure.
std::optional<int64_t> retrieve_ioreg_prop(const std::string &service_class,
                                           const std::string &property_name) {
    // Look for a service matching the supplied class name.
    CFMutableDictionaryRef matching_dict = IOServiceMatching(service_class.c_str());
    if (!matching_dict) {
        return std::nullopt;
    }

#if TARGET_OS_OSX && MAC_OS_X_VERSION_MIN_REQUIRED < 120000 /* MAC_OS_VERSION_12_0 */
    // These are the same variable, just renamed in macOS 12+.
    const mach_port_t kIOMainPortDefault = kIOMasterPortDefault;
#endif
    // IOServiceGetMatchingService consumes a reference to matching_dict, so we don't need
    // to release it ourselves.
    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, matching_dict);
    if (!service) {
        return std::nullopt;
    }

    // Create a CF representation of the registry property of interest.
    const auto cfs_property_name = CFStringCreateWithCString(
            kCFAllocatorDefault, property_name.c_str(), kCFStringEncodingUTF8);
    CFTypeRef property =
            IORegistryEntryCreateCFProperty(service, cfs_property_name, kCFAllocatorDefault, 0);
    IOObjectRelease(service);
    CFRelease(cfs_property_name);
    if (!property) {
        return std::nullopt;
    }
    if (CFGetTypeID(property) == CFNumberGetTypeID()) {
        int64_t value = -1;
        if (!CFNumberGetValue(static_cast<CFNumberRef>(property), kCFNumberSInt64Type, &value)) {
            return std::nullopt;
        }
        return std::make_optional<int64_t>(value);
    }

    // It was not of the expected type.
    return std::nullopt;
}

}  // namespace

namespace dorado::utils {

NS::SharedPtr<MTL::Buffer> create_buffer(MTL::Device *device, size_t length) {
    return NS::TransferPtr(device->newBuffer(length, MTL::ResourceStorageModeShared));
}

NS::SharedPtr<MTL::ComputePipelineState> make_cps(
        MTL::Device *const device,
        const std::string &name,
        const std::vector<std::tuple<std::string, MetalConstant>> &named_constants,
        const std::optional<int> max_total_threads_per_tg) {
    auto metal_kernels = load_kernels(device);

    auto constant_vals = NS::TransferPtr(FunctionConstantValues::alloc()->init());
    for (auto &[cname, constant] : named_constants) {
        const auto ns_name = NS::String::string(cname.c_str(), NS::ASCIIStringEncoding);
        std::visit(overloaded{[&](int val) {
                                  constant_vals->setConstantValue(&val, DataTypeInt, ns_name);
                              },
                              [&](bool val) {
                                  constant_vals->setConstantValue(&val, DataTypeBool, ns_name);
                              },
                              [&](float val) {
                                  constant_vals->setConstantValue(&val, DataTypeFloat, ns_name);
                              }},
                   constant);
    }

    auto kernel_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
    auto kernel = NS::TransferPtr(
            wrap_func_with_err(metal_kernels->newFunction, kernel_name, constant_vals.get()));
    if (!kernel) {
        throw std::runtime_error("Failed to find the kernel: " + name);
    }

    auto cp_descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    cp_descriptor->setComputeFunction(kernel.get());
    if (max_total_threads_per_tg) {
        cp_descriptor->setMaxTotalThreadsPerThreadgroup(*max_total_threads_per_tg);
    }

    auto cps =
            NS::TransferPtr(wrap_func_with_err(device->newComputePipelineState, cp_descriptor.get(),
                                               MTL::PipelineOptionNone, nullptr));
    if (!cps) {
        throw std::runtime_error("Failed to build compute pipeline for " + name);
    }

    return cps;
}

void launch_kernel(ComputePipelineState *const pipeline,
                   CommandQueue *const command_queue,
                   const std::vector<Buffer *> &buffers,
                   const std::vector<int> &tg_buffer_lens,
                   long threadgroups,
                   long threads_per_threadgroup) {
    auto command_buffer = command_queue->commandBuffer();
    launch_kernel_no_wait(pipeline, command_buffer, buffers, tg_buffer_lens, threadgroups,
                          threads_per_threadgroup);

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    auto status = command_buffer->status();
    if (status != MTL::CommandBufferStatusCompleted) {
        spdlog::warn("Synchronous metal command buffer failed: {}", fmt::underlying(status));
    }
}

MTL::CommandBuffer *next_command_buffer(MTL::CommandQueue *queue, int try_count) {
    if (try_count == 0) {
        return queue->commandBuffer();
    }
    // We're on a retry so there must have been an error, so enable additional logging this time around.
    auto descriptor = NS::TransferPtr(MTL::CommandBufferDescriptor::alloc()->init());
    descriptor->setErrorOptions(MTL::CommandBufferErrorOptionEncoderExecutionStatus);
    return queue->commandBuffer(descriptor.get());
}

void launch_kernel_no_wait(ComputePipelineState *const pipeline,
                           CommandBuffer *const command_buffer,
                           const std::vector<Buffer *> &buffers,
                           const std::vector<int> &tg_buffer_lens,
                           long threadgroups,
                           long threads_per_threadgroup) {
    auto compute_encoder = command_buffer->computeCommandEncoder();
    compute_encoder->setComputePipelineState(pipeline);

    // Set up device memory buffers.
    for (auto i = 0; i < (int)buffers.size(); i++) {
        compute_encoder->setBuffer(buffers[i], 0, i);
    }

    // Set lengths of threadgroup memory buffers.
    for (int i = 0; i < (int)tg_buffer_lens.size(); ++i) {
        compute_encoder->setThreadgroupMemoryLength(tg_buffer_lens.at(i), i);
    }

    compute_encoder->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1),
                                          MTL::Size(threads_per_threadgroup, 1, 1));
    compute_encoder->memoryBarrier(BarrierScopeBuffers);
    compute_encoder->endEncoding();
}

bool run_command_buffer(const char *label, MTL::CommandBuffer *cb, int try_count) {
    POINT_OF_INTEREST_SCOPE(metal_utils, run_command_buffer, "label=%s", label);

    name_mtl_object(cb, label);
    cb->commit();
    cb->waitUntilCompleted();

    auto status = cb->status();
    bool success = (status == MTL::CommandBufferStatusCompleted);
    if (success) {
        spdlog::trace("Metal command buffer {}: {} GPU ms {} CPU ms succeeded (try {})", label,
                      1000.f * float(cb->GPUEndTime() - cb->GPUStartTime()),
                      1000.f * float(cb->kernelEndTime() - cb->kernelStartTime()), try_count);
    } else {
        spdlog::warn("Metal command buffer {} failed: status {} (try {})", label,
                     fmt::underlying(status), try_count);
        if (status == MTL::CommandBufferStatusError) {
            report_error(cb->error(), "run_command_buffer");
        }
    }
    return success;
}

static NS::SharedPtr<MTL::Device> mtl_device;

struct MTLAllocator : at::Allocator {
    at::DataPtr allocate(size_t n)
#if !(TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3)
            const
#endif  // < 2.3
            override {
        if (n == 0) {
            return at::DataPtr(nullptr, at::DeviceType::CPU);
        } else if (n >= (size_t(1) << 32)) {
            return at::DataPtr(new char[n], at::DeviceType::CPU);
        }
        auto buffer = mtl_device->newBuffer(n, MTL::ResourceStorageModeShared);
        return at::DataPtr(buffer->contents(), buffer, &deleter, at::DeviceType::CPU);
    }

    static void deleter(void *ptr) { ((MTL::Buffer *)ptr)->release(); }

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
    void copy_data(void *dest, const void *src, std::size_t count) const override {
        default_copy_data(dest, src, count);
    }
#endif  // < 2.3
};
static MTLAllocator mtl_allocator;

NS::SharedPtr<MTL::Device> get_mtl_device() {
    if (!mtl_device) {
        mtl_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        at::SetAllocator(at::DeviceType::CPU, &mtl_allocator);
    }
    return mtl_device;
}

int get_mtl_device_core_count() {
    // We cache the count once it has been obtained.
    static int gpu_core_count = -1;
    if (gpu_core_count != -1) {
        return gpu_core_count;
    }

    // Attempt to directly query the GPU core count.
    if (auto core_count_opt = retrieve_ioreg_prop("AGXAccelerator", "gpu-core-count");
        core_count_opt.has_value()) {
        gpu_core_count = static_cast<int>(core_count_opt.value());
        spdlog::debug("Retrieved GPU core count of {} from IO Registry", gpu_core_count);
        return gpu_core_count;
    }

    // If querying failed, estimate the count based on the Metal device name,
    // with a fallback of 8 (a complete base spec. M1) if it is not recognised.
    gpu_core_count = 8;
    const std::string name = get_mtl_device()->name()->utf8String();
    spdlog::debug("Basing GPU core count on Metal device string {}", name);
    if (name == "Apple M1 Pro") {
        gpu_core_count = 16;
    } else if (name == "Apple M1 Max") {
        gpu_core_count = 32;
    } else if (name == "Apple M1 Ultra") {
        gpu_core_count = 64;
    } else if (name == "Apple M2 GPU" || name == "Apple M4 GPU") {
        // M2 configurations with < 10 cores exist in e.g. MacBook Air, but it's
        // assumed that those configurations would be handled above via IORegistry
        // querying.  The M2 iPad Pro always has 10 GPU cores.  Note also that
        // iOS metal device names in any case appear to have a different form, with
        // "GPU" at the end.
        gpu_core_count = 10;
    }

    spdlog::warn("Failed to retrieve GPU core count from IO Registry: using value of {}",
                 gpu_core_count);
    return gpu_core_count;
}

int get_apple_cpu_perf_core_count() {
    // We cache the count once it has been obtained.
    static int cpu_perf_core_count = -1;
    if (cpu_perf_core_count != -1) {
        return cpu_perf_core_count;
    }

    size_t size = sizeof(cpu_perf_core_count);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cpu_perf_core_count, &size, nullptr, 0) == -1) {
        std::string name = get_mtl_device()->name()->utf8String();
        cpu_perf_core_count = 4;  // Used for M1, M2, and as fallback
        // Lower-spec M1/M2 Pro versions with 6 cores also exist.
        if (name == "Apple M1 Pro" || name == "Apple M1 Max" || name == "Apple M2 Pro" ||
            name == "Apple M2 Max") {
            cpu_perf_core_count = 8;
        } else if (name == "Apple M1 Ultra") {
            cpu_perf_core_count = 16;
        }
        spdlog::warn("Failed to retrieve CPU performance core count from sysctl: using value of {}",
                     cpu_perf_core_count);
    } else {
        spdlog::debug("Retrieved CPU performance core count of {} from sysctl",
                      cpu_perf_core_count);
    }
    return cpu_perf_core_count;
}

MTL::Buffer *mtl_for_tensor(const at::Tensor &x) {
    // Metal kernels assume contiguity.
    if (!x.is_contiguous()) {
        throw std::runtime_error("Tensor is not contiguous");
    }
    auto ptr = (MTL::Buffer *)(x.storage().data_ptr().get_context());
    assert(ptr != nullptr);
    return ptr;
}

NS::SharedPtr<MTL::Buffer> extract_mtl_from_tensor(at::Tensor &&x) {
    auto bfr = NS::RetainPtr(mtl_for_tensor(x));
    x.reset();
    return bfr;
}

ScopedAutoReleasePool::ScopedAutoReleasePool() {
    Class ns_autorelease_pool_class = objc_getClass("NSAutoreleasePool");
    id autorelease_pool_alloc =
            ((id(*)(Class, SEL))objc_msgSend)(ns_autorelease_pool_class, sel_registerName("alloc"));
    m_autorelease_pool =
            ((id(*)(id, SEL))objc_msgSend)(autorelease_pool_alloc, sel_registerName("init"));
}

ScopedAutoReleasePool::~ScopedAutoReleasePool() {
    // Note: This destroys the autorelease pool object itself, along with the objects it is responsible
    // for deleting.
    ((void (*)(id, SEL))objc_msgSend)(m_autorelease_pool, sel_registerName("drain"));
}

size_t get_apple_physical_memory_bytes() {
    size_t mem_size;
    size_t size = sizeof(mem_size);
    if (sysctlbyname("hw.memsize", &mem_size, &size, nullptr, 0) == -1) {
        mem_size = size_t{8} << 30;
        spdlog::warn("Failed to retrieve physical memory size: defaulting to {} bytes", mem_size);
    }
    return mem_size;
}

// MTLCaptureManager can only capture one thing at a time, so guard it with a global.
static std::atomic_bool s_capturing{false};

ScopedMetalCapture::ScopedMetalCapture(MTL::Device *device,
                                       const std::optional<std::filesystem::path> &path) {
    if (s_capturing.exchange(true, std::memory_order_relaxed) != false) {
        spdlog::error("Already performing a GPU capture");
        std::abort();
    }

    // Setup descriptor of what to capture.
    auto descriptor = NS::TransferPtr(MTL::CaptureDescriptor::alloc()->init());
    descriptor->setCaptureObject(device);

    // Dump to a file if requested.
    if (path.has_value()) {
        descriptor->setDestination(MTL::CaptureDestination::CaptureDestinationGPUTraceDocument);
        // Assume that the path is valid UTF-8.
        auto pathstr =
                NS::TransferPtr(NS::String::string(path->string().c_str(), NS::UTF8StringEncoding));
        auto url = NS::TransferPtr(NS::URL::alloc()->initFileURLWithPath(pathstr.get()));
        descriptor->setOutputURL(url.get());
    }

    // Start the capture.
    auto *manager = MTL::CaptureManager::sharedCaptureManager();
    const bool active = wrap_func_with_err(manager->startCapture, descriptor.get());

    if (active) {
        m_device = device;
    } else if (!getenv("MTL_CAPTURE_ENABLED")) {
        // Developer probably forgot to set this, so help them out.
        spdlog::warn("Note that MTL_CAPTURE_ENABLED=1 must be set in order to capture");
    }
}

ScopedMetalCapture::~ScopedMetalCapture() {
    if (m_device) {
        auto *manager = MTL::CaptureManager::sharedCaptureManager();
        manager->stopCapture();
    }
    s_capturing.store(false, std::memory_order_relaxed);
}

void ScopedMetalCapture::inspect_buffer(MTL::Buffer *buffer, MTL::CommandBuffer *cb) {
    if (!m_device) {
        return;
    }

    auto inspector = make_cps(m_device, "inspect_buffer", {}, {});
    launch_kernel_no_wait(inspector.get(), cb, {buffer}, {}, 1, 1);
    if (!run_command_buffer("inspect_buffer", cb, 0)) {
        // Only a warning since it should still be visible in the trace
        spdlog::warn("Buffer inspection failed");
    }
}

}  // namespace dorado::utils
