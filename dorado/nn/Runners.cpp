#include "Runners.h"

#include "decode/CPUDecoder.h"

#if DORADO_GPU_BUILD
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif
#endif  // DORADO_GPU_BUILD

#include <thread>

namespace dorado {

std::pair<std::vector<dorado::Runner>, size_t> create_basecall_runners(
        const dorado::CRFModelConfig& model_config,
        const std::string& device,
        size_t num_runners,
        size_t batch_size,
        size_t chunk_size,
        float memory_fraction,
        bool guard_gpus) {
    std::vector<dorado::Runner> runners;

    // Default is 1 device.  CUDA path may alter this.
    int num_devices = 1;

    if (device == "cpu") {
        num_runners = std::thread::hardware_concurrency();
        if (batch_size == 0) {
            batch_size = 128;
        }
        spdlog::debug("- CPU calling: set batch size to {}, num_runners to {}", batch_size,
                      num_runners);

        for (size_t i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<dorado::ModelRunner<dorado::CPUDecoder>>(
                    model_config, device, chunk_size, batch_size));
        }
    }
#if DORADO_GPU_BUILD
#ifdef __APPLE__
    else if (device == "metal") {
        auto caller = dorado::create_metal_caller(model_config, chunk_size, batch_size);
        for (size_t i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<dorado::MetalModelRunner>(caller));
        }
        if (runners.back()->batch_size() != batch_size) {
            spdlog::debug("- set batch size to {}", runners.back()->batch_size());
        }
    } else {
        throw std::runtime_error(std::string("Unsupported device: ") + device);
    }
#else   // ifdef __APPLE__
    else {
        auto devices = dorado::utils::parse_cuda_device_string(device);
        num_devices = devices.size();
        if (num_devices == 0) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
        for (auto device_string : devices) {
            auto caller = dorado::create_cuda_caller(model_config, chunk_size, batch_size,
                                                     device_string, memory_fraction, guard_gpus);
            for (size_t i = 0; i < num_runners; i++) {
                runners.push_back(std::make_shared<dorado::CudaModelRunner>(caller));
            }
            if (runners.back()->batch_size() != batch_size) {
                spdlog::debug("- set batch size for {} to {}", device_string,
                              runners.back()->batch_size());
            }
        }
    }
#endif  // __APPLE__
#endif  // DORADO_GPU_BUILD

    auto model_stride = runners.front()->model_stride();
    auto adjusted_chunk_size = runners.front()->chunk_size();
    assert(std::all_of(runners.begin(), runners.end(), [&](auto runner) {
        return runner->model_stride() == model_stride &&
               runner->chunk_size() == adjusted_chunk_size;
    }));

    if (chunk_size != adjusted_chunk_size) {
        spdlog::debug("- adjusted chunk size to match model stride: {} -> {}", chunk_size,
                      adjusted_chunk_size);
        chunk_size = adjusted_chunk_size;
    }

    return {runners, num_devices};
}

std::vector<std::unique_ptr<dorado::ModBaseRunner>> create_modbase_runners(
        const std::string& remora_models,
        const std::string& device,
        size_t remora_runners_per_caller,
        size_t remora_batch_size) {
    std::vector<std::filesystem::path> remora_model_list;
    std::istringstream stream{remora_models};
    std::string model;
    while (std::getline(stream, model, ',')) {
        remora_model_list.push_back(model);
    }

    if (remora_model_list.empty()) {
        return {};
    }

    // generate model callers before nodes or it affects the speed calculations
    std::vector<std::unique_ptr<dorado::ModBaseRunner>> remora_runners;
    std::vector<std::string> modbase_devices;
#if DORADO_GPU_BUILD && !defined(__APPLE__)
    if (device != "cpu") {
        modbase_devices = dorado::utils::parse_cuda_device_string(device);
    } else
#endif
    {
        modbase_devices.push_back(device);
    }
    for (const auto& device_string : modbase_devices) {
        auto caller =
                dorado::create_modbase_caller(remora_model_list, remora_batch_size, device_string);
        for (size_t i = 0; i < remora_runners_per_caller; i++) {
            remora_runners.push_back(std::make_unique<dorado::ModBaseRunner>(caller));
        }
    };

    return remora_runners;
}

}  // namespace dorado
