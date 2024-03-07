#include "cuda_utils.h"

#include "math_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/matmul.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <chrono>
#include <exception>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace std::chrono;

namespace dorado::utils {

namespace {

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

std::vector<std::string> parse_cuda_device_string(std::string device_string) {
    std::vector<std::string> devices;
    std::regex e("[0-9]+");
    std::smatch m;

    auto num_devices = torch::cuda::device_count();
    if (device_string.substr(0, 5) != "cuda:") {
        return devices;  // empty vector;
    } else if (device_string == "cuda:all" || device_string == "cuda:auto") {
        for (size_t i = 0; i < num_devices; i++) {
            devices.push_back("cuda:" + std::to_string(i));
        }
    } else {
        while (std::regex_search(device_string, m, e)) {
            for (auto x : m) {
                std::string device_id = x.str();
                int device_idx = std::stoi(device_id);
                if (device_idx >= int(num_devices) || device_idx < 0) {
                    throw std::runtime_error("Invalid CUDA device index \"" + device_id +
                                             "\" from device string " + device_string +
                                             ", there are " + std::to_string(num_devices) +
                                             " visible CUDA devices.");
                }
                devices.push_back("cuda:" + device_id);
            }
            device_string = m.suffix().str();
        }
    }

    return devices;
}

std::vector<CUDADeviceInfo> get_cuda_device_info(std::string device_string) {
    std::vector<CUDADeviceInfo> results;
    std::regex e("[0-9]+");
    std::smatch m;
    auto num_devices = torch::cuda::device_count();

    // Get the set of ids that are in use according to the device_string
    std::set<int> device_ids;
    if (device_string.substr(0, 5) != "cuda:") {
        // Nothing to add to device_ids
    } else if (device_string == "cuda:all" || device_string == "cuda:auto") {
        if (num_devices == 0) {
            throw std::runtime_error("device string set to " + device_string +
                                     " but no CUDA devices available.");
        }
        for (int i = 0; i < int(num_devices); i++) {
            device_ids.insert(i);
        }
    } else {
        while (std::regex_search(device_string, m, e)) {
            for (auto x : m) {
                std::string device_id = x.str();
                int device_idx = std::stoi(device_id);
                if (device_idx >= int(num_devices) || device_idx < 0) {
                    throw std::runtime_error("Invalid CUDA device index \"" + device_id +
                                             "\" from device string " + device_string +
                                             ", there are " + std::to_string(num_devices) +
                                             " visible CUDA devices.");
                }
                device_ids.insert(device_idx);
            }
            device_string = m.suffix().str();
        }
    }

    // Now inspect all the devices on the host to create the CUDADeviceInfo
    for (int device_id = 0; device_id < int(num_devices); device_id++) {
        CUDADeviceInfo device_info;
        device_info.device_id = device_id;

        cudaSetDevice(device_id);
        cudaMemGetInfo(&device_info.free_mem, &device_info.total_mem);
        cudaDeviceGetAttribute(&device_info.compute_cap_major, cudaDevAttrComputeCapabilityMajor,
                               device_id);
        cudaDeviceGetAttribute(&device_info.compute_cap_minor, cudaDevAttrComputeCapabilityMinor,
                               device_id);
        cudaGetDeviceProperties(&device_info.device_properties, device_id);

        device_info.in_use = device_ids.find(device_id) != device_ids.end();

        results.push_back(device_info);
    }

    return results;
}

std::unique_lock<std::mutex> acquire_gpu_lock(int gpu_index, bool use_lock) {
    static std::vector<std::mutex> gpu_mutexes(torch::cuda::device_count());

    return (use_lock ? std::unique_lock<std::mutex>(gpu_mutexes.at(gpu_index))
                     : std::unique_lock<std::mutex>());
}

// This might come in handy for tracking down where big Torch allocations happen
void print_cuda_alloc_info(const std::string &label) {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    auto print_stat_array = [](c10::cuda::CUDACachingAllocator::StatArray &stat,
                               const std::string &lbl) {
        constexpr float gig = 1024.f * 1024.f * 1024.f;
        std::cerr << lbl << "[" << stat[0].current / gig << ", " << stat[0].peak / gig << ", "
                  << stat[0].allocated / gig << ", " << stat[0].freed / gig << "] ";
    };
    std::cerr << "CUDAAlloc cpaf, " << label << " ";
    print_stat_array(stats.allocated_bytes, "All");
    print_stat_array(stats.reserved_bytes, "Rs");
    print_stat_array(stats.active_bytes, "Act");
    print_stat_array(stats.inactive_split_bytes, "In");
    std::cerr << std::endl;
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
