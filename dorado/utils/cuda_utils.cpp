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

#include <chrono>
#include <limits>
#include <regex>
#include <string>
#include <vector>
using namespace std::chrono;

namespace dorado::utils {

void cublas_matmul_f16(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
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
    for (int t = 0; t < timesteps; t++) {
        cublas_matmul_f16(a, b, c);
        host_lstm_step_f16(stream, max_batchsize, layer_size, bias.data_ptr(), c.data_ptr(),
                           state_buf.data_ptr(), a.data_ptr());
    }

    for (int batchsize = min_batchsize; batchsize <= max_batchsize; batchsize += 16) {
        steady_clock::time_point begin = steady_clock::now();
        for (int t = 0; t < timesteps; t++) {
            a = torch::empty({batchsize, layer_size * 2}, options);
            c = torch::empty({batchsize, layer_size * 4}, options);
            cublas_matmul_f16(a, b, c);
            host_lstm_step_f16(stream, batchsize, layer_size, bias.data_ptr(), c.data_ptr(),
                               state_buf.data_ptr(), a.data_ptr());
        }
        steady_clock::time_point end = steady_clock::now();
        float time = duration_cast<nanoseconds>(end - begin).count() / batchsize;

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
    const std::vector<std::vector<int>> batch_sizes = {
            {960, 448, 128},     // 8GB
            {1536, 640, 192},    // 12GB
            {2048, 1024, 256},   // 16GB
            {2048, 1536, 512},   // 24GB
            {2560, 2560, 640},   // 32GB
            {4096, 2560, 1024},  // 40GB
    };

    assert(breakpoints.size() == batch_sizes.size());

    // compute how much free gpu memory and pick the closest breakpoint
    auto available = available_memory(devices);
    int min_available = *std::min_element(available.begin(), available.end()) / 1e+9;
    spdlog::debug("- available GPU memory {}GB", min_available);
    int idx = std::lower_bound(breakpoints.begin(), breakpoints.end(), min_available) -
              breakpoints.begin();
    auto presets = batch_sizes[std::min(idx, static_cast<int>(breakpoints.size() - 1))];

    if (model_path.find("_fast@v") != std::string::npos) {
        return select_batchsize(96, 64, presets[0], devices[0]);
    } else if (model_path.find("_hac@v") != std::string::npos) {
        return select_batchsize(384, 64, presets[1], devices[0]);
    } else if (model_path.find("_sup@v") != std::string::npos) {
        return select_batchsize(1024, 64, presets[2], devices[0]);
    }

    spdlog::warn("> warning: auto batchsize detection failed");
    return 128;
}

}  // namespace dorado::utils
