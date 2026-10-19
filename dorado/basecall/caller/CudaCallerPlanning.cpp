#include "CudaCallerPlanning.h"

#include "config/BasecallModelConfig.h"
#include "decode/Decoder.h"
#include "torch_utils/cuda_utils.h"
#include "utils/log_utils.h"
#include "utils/math_utils.h"

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace dorado::basecall {

namespace {

constexpr float GB = 1.0e9f;

}  // namespace

int cuda_batch_size_granularity(const config::BasecallModelConfig& model_config) {
    // TODO: we may want to use different numbers based on model type and GPU arch
    return model_config.is_tx_model() ? 32 : 64;
}

int64_t cuda_memory_limit(c10::Device device, float memory_limit_fraction) {
    c10::cuda::CUDAGuard device_guard(device);
    c10::cuda::CUDACachingAllocator::emptyCache();
    const int64_t available = utils::available_memory(device);
    DORADO_LOG(debug, "{} memory available: {:.2f}GB", device.str(), available / GB);

    // If running on a Jetson device with unified memory for CPU and GPU we can't use all
    // the available memory for GPU tasks. This way we leave at least half for the CPU,
    // though it's not clear what the ideal split would be.
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    bool is_unified_memory_device = (prop->major == 5 && prop->minor == 3) ||   // TX1
                                    (prop->major == 6 && prop->minor == 2) ||   // TX2
                                    (prop->major == 7 && prop->minor == 2) ||   // Xavier
                                    (prop->major == 8 && prop->minor == 7) ||   // Orin
                                    (prop->major == 11 && prop->minor == 0) ||  // Thor
                                    (prop->major == 12 && prop->minor == 1);    // DGX Spark
    if (is_unified_memory_device) {
        memory_limit_fraction *= 0.5f;
    }

    if (is_unified_memory_device && prop->major >= 8 && available > (32 * GB)) {
        // Restrict Orin and Thor further as there's no benefit to the largest batch sizes
        // and definite downsides to using all the memory.
        memory_limit_fraction *= 0.5f;
    }

    int64_t gpu_mem_limit = static_cast<int64_t>(available * memory_limit_fraction);

    // Allow 1GB for model weights and another 1GB for Torch and CUDA internals.
    gpu_mem_limit -= 2 * GB;

    if (gpu_mem_limit < 0) {
        throw std::runtime_error(
                "Failed to determine safe batch size. Less than 1GB GPU memory available.");
    }
    return gpu_mem_limit;
}

CudaMemoryRequirements cuda_memory_requirements(const config::BasecallModelConfig& model_config) {
    // Determine size of working memory for CRFModel divided by (batch_size * chunk_size).
    int64_t crfmodel_bytes_per_chunk_timestep;
    if (model_config.is_flstm_model()) {
        if (model_config.lstm_size > 1024) {
            DORADO_LOG(warn, "Unexpected model insize {}. Estimating GPU memory requirements.",
                       model_config.lstm_size);
        }
        crfmodel_bytes_per_chunk_timestep = 4096;
    } else if (model_config.out_features.has_value()) {
        const auto out_features = model_config.out_features.value();
        const std::map<int, int64_t> out_features_map{{128, 2312}, {256, 8712}, {4096, 34848}};
        const auto it = out_features_map.upper_bound(out_features - 1);
        if (it == out_features_map.end()) {
            DORADO_LOG(error,
                       "Failed to set GPU memory requirements. Unexpected model out_features {}.",
                       out_features);
            return {0, 0};
        }
        if (it->first != out_features) {
            DORADO_LOG(warn,
                       "Unexpected model out_features {}. Estimating GPU memory requirements.",
                       out_features);
        }
        crfmodel_bytes_per_chunk_timestep = it->second;
    } else {
        const std::map<int, int64_t> insize_map{
                {96, 960}, {128, 1280}, {384, 2816}, {768, 9728}, {1024, 10240}};
        const auto it = insize_map.upper_bound(model_config.lstm_size - 1);
        if (it == insize_map.end()) {
            DORADO_LOG(error, "Failed to set GPU memory requirements. Unexpected model insize {}.",
                       model_config.lstm_size);
            return {0, 0};
        }
        if (it->first != model_config.lstm_size) {
            DORADO_LOG(warn, "Unexpected model insize {}. Estimating GPU memory requirements.",
                       model_config.lstm_size);
        }
        crfmodel_bytes_per_chunk_timestep = it->second;
    }

    const auto decoder_config = decode::create_decoder_config(model_config);
    const int64_t decode_bytes_per_chunk_timestep =
            10 + decode::DecoderOptions{}.beam_width * 4 + decoder_config.num_transitions();

    return {crfmodel_bytes_per_chunk_timestep, decode_bytes_per_chunk_timestep};
}

CudaBatchPlan calculate_cuda_batch_sizes(c10::Device device,
                                         float memory_limit_fraction,
                                         const config::BasecallModelConfig& model_config,
                                         std::optional<PipelineType> pipeline_type,
                                         int requested_batch_size,
                                         bool variable_chunk_sizes) {
    c10::cuda::CUDAGuard device_guard(device);
    c10::cuda::CUDACachingAllocator::emptyCache();
    const int batch_granularity = cuda_batch_size_granularity(model_config);
    const int chunk_granularity = model_config.chunk_size_granularity();
    const int stride = model_config.stride;
    const int min_chunk_size =
            utils::pad_to(model_config.basecaller.overlap() + 1, chunk_granularity);
    auto calculate_T_out = [=](int size) {
        return std::max(min_chunk_size, (size / chunk_granularity) * chunk_granularity) / stride;
    };

    const auto requested_chunk_size = model_config.basecaller.chunk_size();
    std::set<int> T_outs({calculate_T_out(requested_chunk_size)});

    if (pipeline_type == PipelineType::simplex) {
        const char* env_extra_chunk_sizes = std::getenv("DORADO_EXTRA_CHUNK_SIZES");
        if (env_extra_chunk_sizes != nullptr) {
            constexpr char SEPARATOR = ';';
            std::string env_string(env_extra_chunk_sizes);
            for (size_t start = 0, end = 0; end != std::string::npos; start = end + 1) {
                T_outs.insert(calculate_T_out(std::atoi(env_string.c_str() + start)));
                end = env_string.find(SEPARATOR, start);
            }
        } else if (!variable_chunk_sizes) {
            T_outs.insert(calculate_T_out(static_cast<int>(requested_chunk_size * 0.5f)));
        }
    }

    CudaBatchPlan result;
    result.batch_dims.reserve(T_outs.size());
    for (auto iter = T_outs.rbegin(); iter != T_outs.rend(); ++iter) {
        result.batch_dims.push_back({
                .N = batch_granularity,
                .T_in = *iter * stride,
                .T_out = *iter,
        });
    }

    const int64_t gpu_mem_limit = cuda_memory_limit(device, memory_limit_fraction);
    DORADO_LOG(debug, "{} memory limit {:.2f}GB", device.str(), gpu_mem_limit / GB);

    auto [crfmodel_bytes_per_ct, decode_bytes_per_ct] = cuda_memory_requirements(model_config);
    if (crfmodel_bytes_per_ct == 0) {
        return result;
    }

    requested_batch_size = utils::pad_to(requested_batch_size, batch_granularity);
    result.max_batch_sizes.reserve(result.batch_dims.size());
    for (auto& batch_dim : result.batch_dims) {
        const auto bytes_per_chunk =
                (crfmodel_bytes_per_ct + decode_bytes_per_ct) * batch_dim.T_out;
        int max_batch_size = static_cast<int>(gpu_mem_limit / bytes_per_chunk);
        max_batch_size -= max_batch_size % batch_granularity;
        if (max_batch_size < batch_granularity) {
            DORADO_LOG(warn,
                       "{} maximum safe estimated batch size at chunk size {} is only {}. Required "
                       "minimum is {}, GPU may run out of memory.",
                       device.str(), batch_dim.T_in, max_batch_size, batch_granularity);
            max_batch_size = batch_granularity;
        } else {
            DORADO_LOG(debug, "{} maximum safe estimated batch size at chunk size {} is {}",
                       device.str(), batch_dim.T_in, max_batch_size);
        }

        result.max_batch_sizes.push_back(max_batch_size);

        if (requested_batch_size != 0) {
            if (requested_batch_size > max_batch_size) {
                DORADO_LOG(
                        warn,
                        "{}: Requested batch size {} exceeds maximum safe estimated batch size {}.",
                        device.str(), requested_batch_size, max_batch_size);
            }
            batch_dim.N = requested_batch_size;
        }
    }

    return result;
}

int cuda_max_safe_batch_size(c10::Device device,
                             float memory_limit_fraction,
                             const config::BasecallModelConfig& model_config) {
    auto max_batch_sizes = calculate_cuda_batch_sizes(device, memory_limit_fraction, model_config,
                                                      std::nullopt, 0, false)
                                   .max_batch_sizes;
    if (max_batch_sizes.size() != 1) {
        throw std::logic_error(fmt::format("Unexpected count of sizes for {}", device.str()));
    }
    return max_batch_sizes.front();
}

}  // namespace dorado::basecall
