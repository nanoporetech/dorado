#include "cuda_utils.h"

#include "cxxpool.h"
#include "torch/torch.h"

#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include <regex>
#include <string>
#include <vector>

namespace dorado::utils {

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
    for (auto& v : futures) {
        vec.push_back(v.get());
    }
    return vec;
}

int auto_gpu_batch_size(std::string model_path, std::vector<std::string> devices) {
    // memory breakpoints in GB
    const std::vector<int> breakpoints{8, 12, 16, 24, 32};
    // {fast, hac, sup}
    const std::vector<std::vector<int>> batch_sizes = {
            {960, 448, 128},    // 8GB
            {1536, 768, 240},   // 12GB
            {2048, 1024, 320},  // 16GB
            {2048, 1536, 512},  // 24GB
            {2048, 2048, 720}   // 32GB
    };

    assert(breakpoints.size() == batch_sizes.size());

    // compute how much free gpu memory and pick the closest breakpoint
    auto available = available_memory(devices);

    int min_available = *std::min_element(available.begin(), available.end()) / 1e+9;
    int idx = std::lower_bound(breakpoints.begin(), breakpoints.end(), min_available) -
              breakpoints.begin();
    auto presets = batch_sizes[std::min(idx, static_cast<int>(breakpoints.size() - 1))];

    if (model_path.find("_fast@v") != std::string::npos) {
        return presets[0];
    } else if (model_path.find("_hac@v") != std::string::npos) {
        return presets[1];
    } else if (model_path.find("_sup@v") != std::string::npos) {
        return presets[2];
    }

    spdlog::warn("> warning: auto batchsize detection failed");
    return 128;
}

}  // namespace dorado::utils
