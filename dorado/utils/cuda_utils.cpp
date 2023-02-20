#include "cuda_utils.h"

#include "../nn/CRFModel.h"
#include "cxxpool.h"
#include "torch/torch.h"

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
#include <limits>
#include <optional>
#include <regex>
#include <string>
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

    static void check_cuda_result(cudaError_t err) {
        if (err != cudaSuccess) {
            spdlog::error("CUDA event error: {} - {}", cudaGetErrorName(err),
                          cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

public:
    /**
     * Mark the beginning of a profiling section.
     * The timer will start once all previously submitted CUDA work
     * has completed on the active stream.
     */
    void start() { check_cuda_result(cudaEventRecord(m_start)); }

    /**
     * Mark the end of a profiling section.
     * The timer will stop once all previously submitted CUDA work
     * has completed on the active stream.
     */
    void stop() { check_cuda_result(cudaEventRecord(m_stop)); }

    /**
     * Get the time spent on the GPU between the begin and end markers.
     * @note This will block the current CPU thread until the end marker
     * has been reached on the active stream.
     */
    float result_ms() {
        check_cuda_result(cudaEventSynchronize(m_stop));
        float ms = 0;
        check_cuda_result(cudaEventElapsedTime(&ms, m_start, m_stop));
        return ms;
    }

    CUDATimer() {
        check_cuda_result(cudaEventCreate(&m_start));
        check_cuda_result(cudaEventCreate(&m_stop));
    }
    ~CUDATimer() {
        check_cuda_result(cudaEventDestroy(m_start));
        check_cuda_result(cudaEventDestroy(m_stop));
    }
};

}  // namespace

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
    } else if (device_string == "cuda:all") {
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

size_t available_memory(std::string device) {
    size_t free, total;
    cudaSetDevice(std::stoi(device.substr(5)));
    cudaMemGetInfo(&free, &total);
    return free;
}

std::vector<size_t> available_memory(std::vector<std::string> devices) {
    std::vector<size_t> vec;
    cxxpool::thread_pool pool{devices.size()};
    std::vector<std::future<size_t>> futures;
    for (auto device : devices) {
        futures.push_back(pool.push([=] { return available_memory(device); }));
    }
    for (auto &v : futures) {
        vec.push_back(v.get());
    }
    return vec;
}

int select_batchsize(int layer_size, int min_batchsize, int max_batchsize, std::string device) {
    int timesteps = 1000;
    int best_batchsize = max_batchsize;
    float best_time = std::numeric_limits<float>::max();

    c10::cuda::CUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
    auto a = torch::empty({max_batchsize, layer_size * 2}, options);
    auto b = torch::empty({layer_size * 2, layer_size * 4}, options);
    auto c = torch::empty({max_batchsize, layer_size * 4}, options);
    auto bias = torch::empty(layer_size * 4, options);
    auto state_buf = torch::empty({max_batchsize, layer_size}, options);

    // warmup
    for (int t = 0; t < 10; t++) {
        matmul_f16(a, b, c);
        host_lstm_step_f16(stream, max_batchsize, layer_size, bias.data_ptr(), c.data_ptr(),
                           state_buf.data_ptr(), a.data_ptr());
    }

    CUDATimer cuda_timer;
    for (int batchsize = min_batchsize; batchsize <= max_batchsize; batchsize += 16) {
        cuda_timer.start();
        for (int t = 0; t < timesteps; t++) {
            a = torch::empty({batchsize, layer_size * 2}, options);
            c = torch::empty({batchsize, layer_size * 4}, options);
            matmul_f16(a, b, c);
            host_lstm_step_f16(stream, batchsize, layer_size, bias.data_ptr(), c.data_ptr(),
                               state_buf.data_ptr(), a.data_ptr());
        }
        cuda_timer.stop();
        float const time = cuda_timer.result_ms() / batchsize;

        if (time < best_time) {
            best_time = time;
            best_batchsize = batchsize;
        }
    }

    return best_batchsize;
}

int auto_gpu_batch_size(std::string model_path, std::vector<std::string> devices) {
    // memory breakpoints in GB
    const std::vector<int> breakpoints{8, 12, 16, 24, 32, 40};
    // {fast, hac, sup}
    const std::vector<std::array<int, 3>> batch_sizes = {
            {960, 448, 128},     // 8GB
            {1536, 640, 192},    // 12GB
            {2048, 1024, 256},   // 16GB
            {2048, 1536, 512},   // 24GB
            {2560, 2560, 640},   // 32GB
            {4096, 2560, 1024},  // 40GB
    };

    // compute how much free gpu memory and pick the closest breakpoint
    auto available = available_memory(devices);
    int min_available = *std::min_element(available.begin(), available.end()) / 1e+9;
    auto presets = details::try_select_max_batch_sizes(breakpoints, batch_sizes, min_available);
    if (!presets) {
        spdlog::warn(
                "Auto batchsize detection failed. Insufficient memory"
                ", required 8GB, available {}GB",
                min_available);
        return 128;
    }

    if (model_path.find("_fast@v") != std::string::npos) {
        return select_batchsize(96, 64, presets->at(0), devices[0]);
    } else if (model_path.find("_hac@v") != std::string::npos) {
        return select_batchsize(384, 64, presets->at(1), devices[0]);
    } else if (model_path.find("_sup@v") != std::string::npos) {
        return select_batchsize(1024, 64, presets->at(2), devices[0]);
    }

    spdlog::warn("Auto batchsize detection failed");
    return 128;
}

namespace details {
std::optional<std::array<int, 3>> try_select_max_batch_sizes(
        std::vector<int> const &breakpoints,
        std::vector<std::array<int, 3>> const &batch_sizes,
        int available_memory_gb) {
    assert(breakpoints.size() == batch_sizes.size());
    int idx = std::upper_bound(breakpoints.begin(), breakpoints.end(), available_memory_gb) -
              breakpoints.begin() - 1;
    if (idx < 0) {
        return std::nullopt;
    }
    return batch_sizes[idx];
}
}  // namespace details

}  // namespace dorado::utils
