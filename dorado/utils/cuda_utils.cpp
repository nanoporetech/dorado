#include "cuda_utils.h"

#include "cxxpool.h"
#include "math_utils.h"

#include <torch/torch.h>

extern "C" {
#include "koi.h"
}

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include <array>
#include <chrono>
#include <exception>
#include <limits>
#include <optional>
#include <regex>
#include <string>
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

}  // namespace

void matmul_f16(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
    // torch::matmul() is a bit slower than cublasGemmEx() on A100 and half the speed on V100,
    // but an order of magnitude faster on our Windows CI machines (1080 Ti), so dynamically
    // pick which one we should use on first invocation.
    static auto const fastest_mat_mul = [] {
        CUDATimer cuda_timer;

        // Arbitrary tensor lengths to benchmark against.
        // Note: even with sizes this small it still takes ~2s to benchmark cuBLAS on a 1080 Ti.
        const int L = 2048;
        const int M = 192;
        const int N = 384;

        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(c10::kCUDA);
        auto a = torch::empty({L, M}, options);
        auto b = torch::empty({M, N}, options);
        auto c = torch::empty({L, N}, options);

        auto run_N_times = [&](auto matmul_impl) {
            const size_t N = 1000;
            // Warmup then profile
            for (size_t i = 0; i < 10; i++) {
                matmul_impl(a, b, c);
            }
            cuda_timer.start();
            for (size_t i = 0; i < N; i++) {
                matmul_impl(a, b, c);
            }
            cuda_timer.stop();
            return cuda_timer.result_ms();
        };

        float const torch_time = run_N_times(details::matmul_f16_torch);
        float const cublas_time = run_N_times(details::matmul_f16_cublas);
        return cublas_time < torch_time ? details::matmul_f16_cublas : details::matmul_f16_torch;
    }();
    fastest_mat_mul(A, B, C);
}

std::vector<std::string> parse_cuda_device_string(std::string device_string) {
    std::vector<std::string> devices;
    std::regex e("[0-9]+");
    std::smatch m;

    if (device_string.substr(0, 5) != "cuda:") {
        return devices;  // empty vector;
    } else if (device_string == "cuda:all" || device_string == "cuda:auto") {
        auto num_devices = torch::cuda::device_count();
        for (int i = 0; i < num_devices; i++) {
            devices.push_back("cuda:" + std::to_string(i));
        }
    } else {
        while (std::regex_search(device_string, m, e)) {
            for (auto x : m) {
                devices.push_back("cuda:" + x.str());
            }
            device_string = m.suffix().str();
        }
    }

    return devices;
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

int auto_gpu_batch_size(torch::nn::ModuleHolder<torch::nn::AnyModule> module,
                        const dorado::CRFModelConfig &model_config,
                        const DecoderOptions &decoder_options,
                        const torch::TensorOptions &options,
                        int granularity,
                        int chunk_size_in,
                        float memory_limit_fraction) {
#ifdef DORADO_TX2
    return 256;
#else
    int64_t available = available_memory(options.device()) * memory_limit_fraction;
    spdlog::debug("Auto batch size: GPU memory available: {}GB", available / 1.0e9f);

    int64_t crfmodel_bytes_per_nt;
    if (model_config.out_features.has_value()) {
        auto out_features = model_config.out_features.value();
        std::unordered_map<int, int64_t> out_features_map{{128, 2312}, {256, 8712}};
        crfmodel_bytes_per_nt = out_features_map[out_features];
        if (crfmodel_bytes_per_nt == 0) {
            spdlog::warn("Auto batchsize detection failed. Unexpected model out_features {}.",
                         out_features);
            return pad_to(128, granularity);
        }
    } else {
        std::unordered_map<int, int64_t> insize_map{
                {96, 960}, {128, 1280}, {384, 2816}, {768, 9728}, {1024, 10240}};
        crfmodel_bytes_per_nt = insize_map[model_config.insize];
        if (crfmodel_bytes_per_nt == 0) {
            spdlog::warn("Auto batchsize detection failed. Unexpected model insize {}.",
                         model_config.insize);
            return pad_to(128, granularity);
        }
    }

    int64_t decode_bytes_per_nt =
            10 + decoder_options.beam_width * 4 + (1 << (model_config.state_len * 2 + 2));

    auto bytes_per_nt = decode_bytes_per_nt + crfmodel_bytes_per_nt;
    int64_t chunk_size_out = chunk_size_in / model_config.stride;
    available = available - 1.0e9f;  // Allow 1GB for model weights, etc.
    const int max_batch_size = available / (bytes_per_nt * chunk_size_out);

    if (max_batch_size < pad_to(128, granularity) + granularity) {
        spdlog::warn("Auto batchsize detection failed. Estimated max batch size only {}.",
                     max_batch_size);
        return pad_to(128, granularity);
    }

    c10::cuda::CUDAGuard device_guard(options.device());

    int best_batch_size = granularity;
    float best_time = std::numeric_limits<float>::max();
    const int chunk_size = std::min(chunk_size_in, model_config.stride * 300);
    CUDATimer cuda_timer;
    spdlog::debug("Auto batch size: testing up to {} in steps of {}", max_batch_size, granularity);
    for (int batch_size = granularity; batch_size <= max_batch_size; batch_size += granularity) {
        auto input = torch::empty({batch_size, model_config.num_features, chunk_size}, options);

        float time = std::numeric_limits<float>::max();
        for (int i = 0; i < 2; ++i) {  // run twice to eliminate outliers
            cuda_timer.start();
            module->forward(input);
            cuda_timer.stop();
            time = std::min(time, cuda_timer.result_ms() / batch_size);
        }

        spdlog::debug("Auto batchsize: {}, time per chunk {} ms", batch_size, time);
        if (time < best_time) {
            best_time = time;
            best_batch_size = batch_size;
        }
    }

    return best_batch_size;
#endif
}

void handle_cuda_result(int cuda_result) {
    if (cuda_result == cudaSuccess)
        return;

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
void matmul_f16_cublas(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
    constexpr uint16_t HALF_ZERO = 0;      // 0.0 in __half format
    constexpr uint16_t HALF_ONE = 0x3C00;  // 1.0 in __half format
    assert(A.dtype() == torch::kF16 && B.dtype() == torch::kF16 && C.dtype() == torch::kF16);
    assert(A.stride(1) == 1 && B.stride(1) == 1 && C.stride(1) == 1);
    assert(A.size(0) == C.size(0));  // M
    assert(B.size(1) == C.size(1));  // N
    assert(A.size(1) == B.size(0));  // K
    auto res =
            cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, B.size(1),
                         A.size(0), A.size(1), &HALF_ONE, B.data_ptr(), CUDA_R_16F, B.stride(0),
                         A.data_ptr(), CUDA_R_16F, A.stride(0), &HALF_ZERO, C.data_ptr(),
                         CUDA_R_16F, C.stride(0), CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (res != CUBLAS_STATUS_SUCCESS) {
        spdlog::error("CuBLAS error {}", int(res));
        exit(EXIT_FAILURE);
    }
}

void matmul_f16_torch(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
    C.copy_(torch::matmul(A, B));
}

}  // namespace details

}  // namespace dorado::utils
