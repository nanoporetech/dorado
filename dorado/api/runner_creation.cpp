#include "runner_creation.h"

#include "basecall/crf_utils.h"
#include "modbase/ModBaseModelConfig.h"

#if DORADO_GPU_BUILD
#ifdef __APPLE__
#include "basecall/MetalCRFModel.h"
#else
#include "basecall/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif
#endif  // DORADO_GPU_BUILD

#include <cxxpool.h>
#include <spdlog/spdlog.h>

#include <thread>

namespace dorado {

std::pair<std::vector<basecall::RunnerPtr>, size_t> create_basecall_runners(
        const basecall::CRFModelConfig& model_config,
        const std::string& device,
        size_t num_gpu_runners,
        size_t num_cpu_runners,
        size_t batch_size,
        size_t chunk_size,
        float memory_fraction,
        bool guard_gpus) {
#ifdef __APPLE__
    (void)guard_gpus;
#endif

    std::vector<basecall::RunnerPtr> runners;

    // Default is 1 device.  CUDA path may alter this.
    size_t num_devices = 1;

    if (device == "cpu") {
#ifdef DORADO_TX2
        spdlog::warn("CPU basecalling is not supported on this platform. Results may be incorrect");
#endif  // #ifdef DORADO_TX2

        if (batch_size == 0) {
            batch_size = 128;
        }
        if (num_cpu_runners == 0) {
            num_cpu_runners =
                    basecall::auto_calculate_num_runners(model_config, batch_size, memory_fraction);
        }
        spdlog::debug("- CPU calling: set batch size to {}, num_cpu_runners to {}", batch_size,
                      num_cpu_runners);

        for (size_t i = 0; i < num_cpu_runners; i++) {
            runners.push_back(std::make_unique<basecall::ModelRunner>(
                    model_config, device, int(chunk_size), int(batch_size)));
        }
    }
#if DORADO_GPU_BUILD
#ifdef __APPLE__
    else if (device == "metal") {
        auto caller = basecall::create_metal_caller(model_config, int(chunk_size), int(batch_size));
        for (size_t i = 0; i < num_gpu_runners; i++) {
            runners.push_back(std::make_unique<basecall::MetalModelRunner>(caller));
        }
        if (batch_size == 0) {
            spdlog::info(" - set batch size to {}", runners.back()->batch_size());
        } else {
            if (runners.back()->batch_size() != batch_size) {
                spdlog::warn("- set batch size to {}", runners.back()->batch_size());
            }
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

        cxxpool::thread_pool pool{num_devices};
        std::vector<std::shared_ptr<basecall::CudaCaller>> callers;
        std::vector<std::future<std::shared_ptr<basecall::CudaCaller>>> futures;

        for (auto device_string : devices) {
            futures.push_back(pool.push(basecall::create_cuda_caller, model_config, int(chunk_size),
                                        int(batch_size), device_string, memory_fraction,
                                        guard_gpus));
        }

        for (auto& caller : futures) {
            callers.push_back(caller.get());
        }

        for (size_t j = 0; j < num_devices; j++) {
            for (size_t i = 0; i < num_gpu_runners; i++) {
                runners.push_back(std::make_unique<basecall::CudaModelRunner>(callers[j]));
            }
            if (batch_size == 0) {
                spdlog::info(" - set batch size for {} to {}", devices[j],
                             runners.back()->batch_size());
            } else {
                if (runners.back()->batch_size() != batch_size) {
                    spdlog::warn("- set batch size for {} to {}", devices[j],
                                 runners.back()->batch_size());
                }
            }
        }
    }
#endif  // __APPLE__
#else   // DORADO_GPU_BUILD
    (void)num_gpu_runners;
#endif  // DORADO_GPU_BUILD

#ifndef NDEBUG
    auto model_stride = runners.front()->model_stride();
#endif
    auto adjusted_chunk_size = runners.front()->chunk_size();
    assert(std::all_of(runners.begin(), runners.end(), [&](const auto& runner) {
        return runner->model_stride() == model_stride &&
               runner->chunk_size() == adjusted_chunk_size;
    }));

    if (chunk_size != adjusted_chunk_size) {
        spdlog::debug("- adjusted chunk size to match model stride: {} -> {}", chunk_size,
                      adjusted_chunk_size);
        chunk_size = adjusted_chunk_size;
    }

    return {std::move(runners), num_devices};
}

std::vector<modbase::RunnerPtr> create_modbase_runners(
        const std::vector<std::filesystem::path>& remora_models,
        const std::string& device,
        size_t remora_runners_per_caller,
        size_t remora_batch_size) {
    if (remora_models.empty()) {
        return {};
    }

    modbase::check_modbase_multi_model_compatibility(remora_models);

    // generate model callers before nodes or it affects the speed calculations
    std::vector<modbase::RunnerPtr> remora_runners;
    std::vector<std::string> modbase_devices;

    int remora_callers = 1;
    if (device == "cpu") {
        modbase_devices.push_back(device);
        remora_batch_size = 128;
        remora_runners_per_caller = 1;
        remora_callers = std::thread::hardware_concurrency();
    }
#if DORADO_GPU_BUILD
#ifdef __APPLE__
    else if (device == "metal") {
        modbase_devices.push_back(device);
    }
#else   // ifdef __APPLE__
    else {
        modbase_devices = dorado::utils::parse_cuda_device_string(device);
    }
#endif  // __APPLE__
#endif  // DORADO_GPU_BUILD
    for (const auto& device_string : modbase_devices) {
        for (int i = 0; i < remora_callers; ++i) {
            auto caller = modbase::create_modbase_caller(remora_models, int(remora_batch_size),
                                                         device_string);
            for (size_t j = 0; j < remora_runners_per_caller; j++) {
                remora_runners.push_back(std::make_unique<modbase::ModBaseRunner>(caller));
            }
        }
    };

    return remora_runners;
}

}  // namespace dorado
