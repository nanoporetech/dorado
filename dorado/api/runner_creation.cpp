#include "runner_creation.h"

#include "basecall/ModelRunner.h"
#include "basecall/crf_utils.h"
#include "config/ModBaseModelConfig.h"

#if DORADO_METAL_BUILD
#include "basecall/MetalModelRunner.h"
#elif DORADO_CUDA_BUILD
#include "basecall/CudaCaller.h"
#include "basecall/CudaModelRunner.h"
#include "torch_utils/cuda_utils.h"
#endif

#include <cxxpool.h>
#include <spdlog/spdlog.h>

#include <thread>

namespace dorado::api {

std::pair<std::vector<basecall::RunnerPtr>, size_t> create_basecall_runners(
        basecall::BasecallerCreationParams params,
        size_t num_gpu_runners,
        size_t num_cpu_runners) {
    std::vector<basecall::RunnerPtr> runners;

    // Default is 1 device.  CUDA path may alter this.
    size_t num_devices = 1;

    if (params.device == "cpu") {
#if DORADO_TX2
        spdlog::warn("CPU basecalling is not supported on this platform. Results may be incorrect");
#endif  // DORADO_TX2

        if (num_cpu_runners == 0) {
            num_cpu_runners = basecall::auto_calculate_num_runners(params.model_config,
                                                                   params.memory_limit_fraction);
        }
        spdlog::debug("- CPU calling: set num_cpu_runners to {}", num_cpu_runners);
        for (size_t i = 0; i < num_cpu_runners; i++) {
            runners.push_back(
                    std::make_unique<basecall::ModelRunner>(params.model_config, params.device));
        }
        if (runners.back()->batch_size() != (size_t)params.model_config.basecaller.batch_size()) {
            spdlog::debug("- CPU calling: set batch_size to {}", runners.back()->batch_size());
        }
    }
#if DORADO_METAL_BUILD
    else if (params.device == "metal") {
        auto caller = create_metal_caller(params.model_config, params.memory_limit_fraction);
        for (size_t i = 0; i < num_gpu_runners; i++) {
            runners.push_back(std::make_unique<basecall::MetalModelRunner>(caller));
        }
        if (params.model_config.basecaller.batch_size() == 0) {
            spdlog::info(" - set batch size to {}", runners.back()->batch_size());
        } else if (runners.back()->batch_size() !=
                   (size_t)params.model_config.basecaller.batch_size()) {
            spdlog::warn("- set batch size to {}", runners.back()->batch_size());
        }
    }
#endif  // DORADO_METAL_BUILD
#if DORADO_CUDA_BUILD
    else {
        auto devices = dorado::utils::parse_cuda_device_string(params.device);
        num_devices = devices.size();
        if (num_devices == 0) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }

        cxxpool::thread_pool pool{num_devices};
        std::vector<std::shared_ptr<basecall::CudaCaller>> callers;
        std::vector<std::future<std::shared_ptr<basecall::CudaCaller>>> futures;

        futures.reserve(devices.size());

        for (const auto& device_string : devices) {
            basecall::BasecallerCreationParams per_device_params = {
                    params.model_config,
                    device_string,
                    params.memory_limit_fraction,
                    params.pipeline_type,
                    params.batch_size_time_penalty,
                    params.run_batchsize_benchmarks,
                    params.emit_batchsize_benchmarks};
            futures.push_back(pool.push(create_cuda_caller, per_device_params));
        }

        callers.reserve(futures.size());
        for (auto& caller : futures) {
            callers.push_back(caller.get());
        }

        for (size_t j = 0; j < num_devices; j++) {
            size_t num_batch_dims = callers[j]->num_batch_dims();
            for (size_t i = 0; i < num_gpu_runners; i++) {
                for (size_t batch_dims_idx = 0; batch_dims_idx < num_batch_dims; ++batch_dims_idx) {
                    runners.push_back(std::make_unique<basecall::CudaModelRunner>(callers[j],
                                                                                  batch_dims_idx));
                }
            }
        }
    }
#else
    else {
        throw std::runtime_error("Unsupported device: " + params.device);
    }
    (void)num_gpu_runners;
#endif

    return {std::move(runners), num_devices};
}

std::vector<modbase::RunnerPtr> create_modbase_runners(
        const std::vector<std::filesystem::path>& modbase_models,
        const std::string& device,
        size_t runners_per_caller,
        size_t batch_size) {
    if (modbase_models.empty()) {
        return {};
    }

    config::check_modbase_multi_model_compatibility(modbase_models);

    // generate model callers before nodes or it affects the speed calculations
    std::vector<modbase::RunnerPtr> runners;
    std::vector<std::string> modbase_devices;

    int num_callers = 1;
    if (device == "cpu") {
        modbase_devices.push_back(device);
        batch_size = 128;
        runners_per_caller = 1;
        num_callers = std::thread::hardware_concurrency();
    }
#if DORADO_METAL_BUILD
    else if (device == "metal") {
        modbase_devices.push_back(device);
    }
#elif DORADO_CUDA_BUILD
    else {
        modbase_devices = dorado::utils::parse_cuda_device_string(device);
    }
#endif
    for (const auto& device_string : modbase_devices) {
        for (int i = 0; i < num_callers; ++i) {
            auto caller = create_modbase_caller(modbase_models, int(batch_size), device_string);
            for (size_t j = 0; j < runners_per_caller; j++) {
                runners.push_back(std::make_unique<modbase::ModBaseRunner>(caller));
            }
        }
    };

    return runners;
}

#if DORADO_CUDA_BUILD
size_t get_num_batch_dims(const std::shared_ptr<basecall::CudaCaller>& caller) {
    return caller->num_batch_dims();
}
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::CudaCaller> caller,
                                           size_t batch_dims_idx) {
    return std::make_unique<basecall::CudaModelRunner>(std::move(caller), batch_dims_idx);
}
#elif DORADO_METAL_BUILD
size_t get_num_batch_dims(const std::shared_ptr<basecall::MetalCaller>&) {
    return 1;  // Always 1 for Metal. Just needed for a unified interface for GPU builds.
}
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::MetalCaller> caller, size_t) {
    return std::make_unique<basecall::MetalModelRunner>(std::move(caller));
}
#endif

modbase::RunnerPtr create_modbase_runner(std::shared_ptr<modbase::ModBaseCaller> caller) {
    return std::make_unique<::dorado::modbase::ModBaseRunner>(std::move(caller));
}

}  // namespace dorado::api
