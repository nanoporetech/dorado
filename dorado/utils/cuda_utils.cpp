#include "cuda_utils.h"

#include "torch/torch.h"

#include <cuda_runtime_api.h>

#include <regex>
#include <string>
#include <vector>

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

size_t available_memory(std::vector<std::string> devices) {
    size_t free, total;
    std::vector<size_t> vec;

    for (auto device : devices) {
        cudaSetDevice(std::stoi(device.substr(5)));
        cudaMemGetInfo(&free, &total);
        vec.push_back(free);
    }

    return *std::min_element(vec.begin(), vec.end());
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
    int available = available_memory(devices) / (1024 * 1024 * 1024);
    int idx = std::lower_bound(breakpoints.begin(), breakpoints.end(), available) -
              breakpoints.begin();
    auto presets = batch_sizes[std::min(idx, static_cast<int>(breakpoints.size() - 1))];

    if (model_path.find("_fast@v") != std::string::npos) {
        return presets[0];
    } else if (model_path.find("_hac@v") != std::string::npos) {
        return presets[1];
    } else if (model_path.find("_sup@v") != std::string::npos) {
        return presets[2];
    }

    std::cerr << "> warning: auto batchsize detection failed" << std::endl;
    return 128;
}
