#include "cuda_utils.h"

#include "torch_utils/gpu_monitor.h"
#include "utils/PostCondition.h"
#include "utils/math_utils.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <exception>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std::chrono;

namespace dorado::utils {

namespace {
const std::string_view USAGE_HELP{"CUDA device string format: \"cuda:0,...,N\" or \"cuda:all\"."};

/**
 * Wrapper around CUDA events to measure GPU timings.
 */
class CUDATimer {
    cudaEvent_t m_start, m_stop;

    CUDATimer(const CUDATimer &) = delete;
    CUDATimer &operator=(const CUDATimer &) = delete;

public:
    /**
     * Mark the beginning of a profiling section.
     * The timer will start once all previously submitted CUDA work
     * has completed on the active stream.
     */
    void start() { handle_cuda_result(cudaEventRecord(m_start)); }

    /**
     * Mark the end of a profiling section.
     * The timer will stop once all previously submitted CUDA work
     * has completed on the active stream.
     */
    void stop() { handle_cuda_result(cudaEventRecord(m_stop)); }

    /**
     * Get the time spent on the GPU between the begin and end markers.
     * @note This will block the current CPU thread until the end marker
     * has been reached on the active stream.
     */
    float result_ms() {
        handle_cuda_result(cudaEventSynchronize(m_stop));
        float ms = 0;
        handle_cuda_result(cudaEventElapsedTime(&ms, m_start, m_stop));
        return ms;
    }

    CUDATimer() {
        handle_cuda_result(cudaEventCreate(&m_start));
        handle_cuda_result(cudaEventCreate(&m_stop));
    }
    ~CUDATimer() {
        handle_cuda_result(cudaEventDestroy(m_start));
        handle_cuda_result(cudaEventDestroy(m_stop));
    }
};

enum class MatmulMode { TORCH, CUBLAS };

MatmulMode get_cuda_matmul_fp16_mode() {
    const char *env_matmul_fp16_mode = std::getenv("DORADO_MATMUL_FP16_MODE");
    if (env_matmul_fp16_mode != nullptr) {
        std::string_view matmul_fp16_mode_str(env_matmul_fp16_mode);
        spdlog::debug("> Found DORADO_MATMUL_FP16_MODE={}", matmul_fp16_mode_str);
        if (matmul_fp16_mode_str == "TORCH") {
            spdlog::debug(">   Using torch::matmul");
            return MatmulMode::TORCH;
        } else if (matmul_fp16_mode_str == "CUBLAS") {
            spdlog::debug(">   Using cublasGemmEx");
            return MatmulMode::CUBLAS;
        }
        spdlog::debug(">   Ignoring unrecognized option. Select from TORCH or CUBLAS.");
    }

    // torch::matmul() is a bit slower than cublasGemmEx() on A100 and V100, and 2x slower on TX2
    // but an order of magnitude faster on 1080 Ti (sm61)
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    bool is_sm61 = (prop->major == 6 && prop->minor == 1);
    if (is_sm61) {
        return MatmulMode::TORCH;
    }
    return MatmulMode::CUBLAS;
}

}  // namespace

void matmul_f16(const at::Tensor &A, const at::Tensor &B, at::Tensor &C) {
    static const auto selected_mat_mul = [] {
        switch (get_cuda_matmul_fp16_mode()) {
        case MatmulMode::TORCH:
            return details::matmul_f16_torch;
        case MatmulMode::CUBLAS:
            return details::matmul_f16_cublas;
        default:
            throw std::logic_error("Unknown MATMUL_FP16 mode");
        }
    }();
    selected_mat_mul(A, B, C);
}

namespace {

bool is_cuda_device_string(const std::string &device_string) {
    return device_string.substr(0, 5) == "cuda:";
}

std::optional<int> try_get_index(const std::string &index_string) {
    int index{};
    try {
        index = std::stoi(index_string);
    } catch (std::exception &) {
        return std::nullopt;
    }

    // std::stoi is more forgiving than we require, confirm it has no extraneous characters
    if (std::to_string(index) != index_string) {
        return std::nullopt;
    }

    return index;
}

bool validate_device_id_in_range(const std::string &device_string,
                                 const int device_id,
                                 const std::size_t num_devices,
                                 std::string &error_message) {
    if (device_id < static_cast<int>(num_devices) && device_id >= 0) {
        return true;
    }
    std::ostringstream oss{};
    oss << "Invalid CUDA device index '" << device_id << "' from device string "
        << std::quoted(device_string) << ", there are " << num_devices << " visible CUDA devices.";
    error_message = oss.str();
    return false;
}

bool try_add_unique_id(const std::string &device_string,
                       const int device_id,
                       std::set<int> &unique_ids,
                       std::string &error_message) {
    auto [_, inserted] = unique_ids.insert(device_id);
    if (inserted) {
        return true;
    }
    std::ostringstream oss{};
    oss << "Duplicate device index '" << device_id << "' from device string "
        << std::quoted(device_string) << ".";
    error_message = oss.str();
    return false;
}

bool try_add_all_devices(const std::string &device_string,
                         const std::size_t num_devices,
                         std::vector<int> &device_ids,
                         std::string &error_message) {
    if (num_devices == 0) {
        error_message = "device string set to " + device_string + " but no CUDA devices available.";
        return false;
    }
    for (size_t i = 0; i < num_devices; ++i) {
        device_ids.push_back(static_cast<int>(i));
    }
    return true;
}

bool try_parse_id_list(const std::string &device_string,
                       const std::size_t num_devices,
                       std::vector<int> &device_ids,
                       std::string &error_message) {
    std::set<int> unique_ids{};
    std::stringstream id_stream(device_string.substr(5));
    std::string token{};
    while (std::getline(id_stream, token, ',')) {
        auto device_id = try_get_index(token);
        if (!device_id) {
            error_message = "Invalid CUDA device string: " + device_string;
            return false;
        }
        if (!validate_device_id_in_range(device_string, *device_id, num_devices, error_message)) {
            return false;
        }
        if (!try_add_unique_id(device_string, *device_id, unique_ids, error_message)) {
            return false;
        }
    }

    if (unique_ids.empty()) {
        std::ostringstream oss{};
        oss << "No device index found in CUDA device string " << std::quoted(device_string) << ".";
        error_message = oss.str();
        return false;
    }
    device_ids = std::vector(unique_ids.begin(), unique_ids.end());

    return true;
}

}  // namespace

bool try_parse_cuda_device_string(const std::string &device_string,
                                  std::vector<std::string> &devices,
                                  std::string &error_message) {
    std::vector<int> device_ids{};

    if (!details::try_parse_device_ids(device_string, utils::gpu_monitor::get_device_count(),
                                       device_ids, error_message)) {
        return false;
    }

    for (const auto device_id : device_ids) {
        devices.push_back("cuda:" + std::to_string(device_id));
    }
    return true;
}

std::vector<std::string> parse_cuda_device_string(const std::string &device_string) {
    std::vector<std::string> devices{};
    std::string error_message{};
    if (!try_parse_cuda_device_string(device_string, devices, error_message)) {
        throw std::runtime_error(error_message);
    }

    return devices;
}

std::vector<CUDADeviceInfo> get_cuda_device_info(const std::string &device_string,
                                                 bool include_unused) {
    const auto num_devices = utils::gpu_monitor::get_device_count();
    std::string error_message{};
    std::vector<int> requested_device_ids{};
    if (!details::try_parse_device_ids(device_string, num_devices, requested_device_ids,
                                       error_message)) {
        throw std::runtime_error(error_message);
    }

    // Now inspect all the devices on the host to create the CUDADeviceInfo
    std::vector<CUDADeviceInfo> results;
    for (int device_id = 0; device_id < int(num_devices); device_id++) {
        CUDADeviceInfo device_info;
        device_info.device_id = device_id;
        device_info.in_use = std::find(requested_device_ids.begin(), requested_device_ids.end(),
                                       device_id) != requested_device_ids.end();
        if (!include_unused && !device_info.in_use) {
            continue;
        }

        cudaSetDevice(device_id);
        cudaMemGetInfo(&device_info.free_mem, &device_info.total_mem);
        cudaDeviceGetAttribute(&device_info.compute_cap_major, cudaDevAttrComputeCapabilityMajor,
                               device_id);
        cudaDeviceGetAttribute(&device_info.compute_cap_minor, cudaDevAttrComputeCapabilityMinor,
                               device_id);
        cudaGetDeviceProperties(&device_info.device_properties, device_id);

        if (!device_info.in_use) {
            cudaDeviceReset();
        }
        results.push_back(device_info);
    }

    return results;
}

std::string get_cuda_gpu_names(const std::string &device_string) {
    auto dev_info = utils::get_cuda_device_info(device_string, false);  // ignore unused GPUs
    std::set<std::string> gpu_strs;
    std::string gpu_names;

    for (const auto &dev : dev_info) {
        gpu_strs.insert(dev.device_properties.name);
    }

    for (const auto &gpu_id : gpu_strs) {
        if (!gpu_names.empty()) {
            gpu_names += "|";
        }
        gpu_names += gpu_id;
    }

    return gpu_names;
}

std::unique_lock<std::mutex> acquire_gpu_lock(int gpu_index, bool use_lock) {
    static std::vector<std::mutex> gpu_mutexes(torch::cuda::device_count());

    return (use_lock ? std::unique_lock<std::mutex>(gpu_mutexes.at(gpu_index))
                     : std::unique_lock<std::mutex>());
}

// This might come in handy for tracking down where big Torch allocations happen
void print_cuda_alloc_info(const std::string &label) {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    auto print_stat_array = [](const auto &stat, const std::string &lbl) {
        constexpr float gig = 1024.f * 1024.f * 1024.f;
        std::cerr << lbl << "[" << stat[0].current / gig << ", " << stat[0].peak / gig << ", "
                  << stat[0].allocated / gig << ", " << stat[0].freed / gig << "] ";
    };
    std::cerr << "CUDAAlloc cpaf, " << label << " ";
    print_stat_array(stats.allocated_bytes, "All");
    print_stat_array(stats.reserved_bytes, "Rs");
    print_stat_array(stats.active_bytes, "Act");
    print_stat_array(stats.inactive_split_bytes, "In");
    std::cerr << '\n' << std::flush;
}

// Note that in general the torch caching allocator may be consuming
// significant memory that could be freed if required.
size_t available_memory(torch::Device device) {
    size_t free, total;
    c10::cuda::CUDAGuard device_guard(device);
    cudaMemGetInfo(&free, &total);
    return free;
}

void handle_cuda_result(int cuda_result) {
    if (cuda_result == cudaSuccess) {
        return;
    }

    if (cuda_result == cudaErrorNoKernelImageForDevice) {
        throw std::runtime_error(
                std::string("Dorado cannot support the CUDA device being used,"
                            " as the compute capability version is incompatible."));
    } else {
        throw std::runtime_error(std::string("Cuda error: ") +
                                 cudaGetErrorString(cudaError_t(cuda_result)));
    }
}

namespace details {

bool try_parse_device_ids(const std::string &device_string,
                          const std::size_t num_devices,
                          std::vector<int> &device_ids,
                          std::string &error_message) {
    auto append_help_to_error_message = utils::PostCondition([&error_message] {
        if (!error_message.empty()) {
            error_message += "\n" + std::string{USAGE_HELP};
        }
    });
    if (!is_cuda_device_string(device_string)) {
        // Not an error as there are valid non cuda device strings, e.g. "cpu".
        device_ids = {};
        return true;
    }

    if (device_string == "cuda:all" || device_string == "cuda:auto") {
        return try_add_all_devices(device_string, num_devices, device_ids, error_message);
    }

    return try_parse_id_list(device_string, num_devices, device_ids, error_message);
}

void matmul_f16_cublas(const at::Tensor &A, const at::Tensor &B, at::Tensor &C) {
    constexpr uint16_t HALF_ZERO = 0;      // 0.0 in __half format
    constexpr uint16_t HALF_ONE = 0x3C00;  // 1.0 in __half format
    assert(A.dtype() == torch::kF16 && B.dtype() == torch::kF16 && C.dtype() == torch::kF16);
    assert(A.stride(1) == 1 && B.stride(1) == 1 && C.stride(1) == 1);
    assert(A.size(0) == C.size(0));  // M
    assert(B.size(1) == C.size(1));  // N
    assert(A.size(1) == B.size(0));  // K
    auto res = cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                            int(B.size(1)), int(A.size(0)), int(A.size(1)), &HALF_ONE, B.data_ptr(),
                            CUDA_R_16F, int(B.stride(0)), A.data_ptr(), CUDA_R_16F,
                            int(A.stride(0)), &HALF_ZERO, C.data_ptr(), CUDA_R_16F,
                            int(C.stride(0)), CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (res != CUBLAS_STATUS_SUCCESS) {
        spdlog::error("CuBLAS error {}", int(res));
        exit(EXIT_FAILURE);
    }
}

void matmul_f16_torch(const at::Tensor &A, const at::Tensor &B, at::Tensor &C) {
    torch::matmul_out(C, A, B);
}

}  // namespace details

}  // namespace dorado::utils
