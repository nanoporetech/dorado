#include "metal_utils.h"

#include <CoreFoundation/CoreFoundation.h>
#if !TARGET_OS_IPHONE
#include <IOKit/IOKitLib.h>
#endif
#include <Metal/Metal.hpp>
#include <mach-o/dyld.h>
#include <spdlog/spdlog.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>
#include <torch/torch.h>

#include <filesystem>

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

NS::String *get_library_location() {
    char ns_path[PATH_MAX + 1];
    uint32_t size = sizeof(ns_path);
    _NSGetExecutablePath(ns_path, &size);

    fs::path exepth{ns_path};
    fs::path mtllib{"../lib/default.metallib"};
    fs::path fspath = exepth.parent_path() / mtllib;

    return NS::String::string(fspath.c_str(), NS::ASCIIStringEncoding);
}

// Returns an ASCII std::string associated with the given CFStringRef.
std::string cfstringref_to_string(const CFStringRef cfstringref) {
    // There does exist an API to directly return a char* pointer, but this is documented as
    // failing on an arbitrary basis, and did fail empirically.
    const auto utf16_len = CFStringGetLength(cfstringref);
    // We must leave room the for zero terminator, or CFStringGetCString will fail.
    const auto max_ascii_len =
            CFStringGetMaximumSizeForEncoding(utf16_len, kCFStringEncodingASCII) + 1;
    // CFStringGetCString wants to supply its own zero terminator, so write to an intermediate
    // buffer used for constructing the final std::string.
    std::vector<char> buffer(max_ascii_len);
    if (CFStringGetCString(cfstringref, &buffer[0], buffer.size(), kCFStringEncodingASCII)) {
        return std::string(buffer.data());
    }

    return std::string("");
}

#if !TARGET_OS_IPHONE

// Retrieves a dictionary of int64_t properties associated with a given service/property.
// Returns true on success.
bool retrieve_ioreg_props(const std::string &service_class,
                          const std::string &property_name,
                          std::unordered_map<std::string, int64_t> &props) {
    // Look for a service matching the supplied class name.
    CFMutableDictionaryRef matching_dict = IOServiceMatching(service_class.c_str());
    if (!matching_dict) {
        return false;
    }
    // Note: kIOMainPortDefault was introduced on MacOS 12.  If support for earlier versions
    // is needed an alternate constant will be needed.
    // IOServiceGetMatchingService consumes a reference to matching_dict, so we don't need
    // to release it ourselves.
    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, matching_dict);
    if (!service) {
        return false;
    }

    // Create a CF representation of the registry property of interest.
    const auto cfs_property_name = CFStringCreateWithCString(
            kCFAllocatorDefault, property_name.c_str(), kCFStringEncodingUTF8);
    CFTypeRef property =
            IORegistryEntryCreateCFProperty(service, cfs_property_name, kCFAllocatorDefault, 0);
    IOObjectRelease(service);
    CFRelease(cfs_property_name);
    if (!property) {
        return false;
    }
    if (CFGetTypeID(property) != CFDictionaryGetTypeID()) {
        CFRelease(property);
        return false;
    }

    // Retrieve entries with integer keys from the CFDictionary we constructed.

    // No implicit conversion from lambda to function pointer if it captures, so
    // just use the context parameter to point to the unordered_map being populated.
    const auto process_kvs = [](CFTypeRef key_ref, CFTypeRef value_ref, void *ctx) {
        auto props_ptr = static_cast<std::unordered_map<std::string, int64_t> *>(ctx);
        // Presumably keys are always strings -- ignore anything that isn't.
        // Also ignore non-integer values, of which there are examples that are not
        // currently relevant.
        if (CFGetTypeID(key_ref) != CFStringGetTypeID() ||
            CFGetTypeID(value_ref) != CFNumberGetTypeID())
            return;
        const std::string key = cfstringref_to_string(static_cast<CFStringRef>(key_ref));
        int64_t value = -1;
        if (!CFNumberGetValue(static_cast<CFNumberRef>(value_ref), kCFNumberSInt64Type, &value)) {
            return;
        }
        props_ptr->insert({key, value});
    };

    CFDictionaryApplyFunction(static_cast<CFDictionaryRef>(property), process_kvs, &props);
    CFRelease(property);

    return true;
}

#endif  // if !TARGET_OS_IPHONE

}  // namespace

namespace dorado::utils {

MTL::Buffer *create_buffer(MTL::Device *device, size_t length) {
    return device->newBuffer(length, MTL::ResourceStorageModeShared);
}

MTL::ComputePipelineState *make_cps(
        MTL::Device *const device,
        const std::string &name,
        const std::vector<std::tuple<std::string, MetalConstant>> &named_constants,
        const int max_total_threads_per_tg) {
    NS::Error *error;
    auto default_library = device->newDefaultLibrary();

    if (!default_library) {
        auto lib_path = get_library_location();
        default_library = device->newLibrary(lib_path, &error);
        if (!default_library) {
            throw std::runtime_error("Failed to load metallib library.");
        }
    }

    auto constant_vals = FunctionConstantValues::alloc();
    constant_vals->init();
    for (auto &[name, constant] : named_constants) {
        const auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
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
    auto kernel = default_library->newFunction(kernel_name, constant_vals, &error);
    CFRelease(constant_vals);
    if (!kernel) {
        throw std::runtime_error("Failed to find the kernel: " + name);
    }

    auto cp_descriptor = MTL::ComputePipelineDescriptor::alloc();
    cp_descriptor->init();
    cp_descriptor->setComputeFunction(kernel);
    if (max_total_threads_per_tg != -1)
        cp_descriptor->setMaxTotalThreadsPerThreadgroup(max_total_threads_per_tg);

    auto cps = device->newComputePipelineState(cp_descriptor, MTL::PipelineOptionNone, nullptr,
                                               &error);
    CFRelease(cp_descriptor);
    if (cps == nullptr) {
        auto e_code = std::to_string(((int)error->code()));
        auto e_str = error->domain()->cString(NS::ASCIIStringEncoding);
        throw std::runtime_error("failed to build compute pipeline for " + name + " - " + e_str +
                                 ": error " + e_code);
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
    for (int i = 0; i < tg_buffer_lens.size(); ++i) {
        compute_encoder->setThreadgroupMemoryLength(tg_buffer_lens.at(i), i);
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

MTL::Device *get_mtl_device() {
    if (mtl_device == nullptr) {
        mtl_device = MTL::CreateSystemDefaultDevice();
        torch::SetAllocator(torch::DeviceType::CPU, &mtl_allocator);
    }
    return mtl_device;
}

int get_mtl_device_core_count() {
    // We cache the count once it has been obtained.
    static int gpu_core_count = -1;
    if (gpu_core_count != -1)
        return gpu_core_count;

#if !TARGET_OS_IPHONE
    // Attempt to directly query the GPU core count.
    std::unordered_map<std::string, int64_t> gpu_specs;
    if (retrieve_ioreg_props("AGXAccelerator", "GPUConfigurationVariable", gpu_specs)) {
        if (auto gpu_cores_it = gpu_specs.find("num_cores"); gpu_cores_it != gpu_specs.cend()) {
            gpu_core_count = gpu_cores_it->second;
            spdlog::debug("Retrieved GPU core count of {} from IO Registry", gpu_core_count);
            return gpu_core_count;
        }
    }
#endif  // if !TARGET_OS_IPHONE

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
    } else if (name == "Apple M2 GPU") {
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
    if (cpu_perf_core_count != -1)
        return cpu_perf_core_count;

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

int auto_gpu_batch_size() { return 48 * get_mtl_device_core_count(); }

}  // namespace dorado::utils
