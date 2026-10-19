#include "MetalCallerPlanning.h"

#include "config/BasecallModelConfig.h"
#include "decode/Decoder.h"
#include "torch_utils/metal_utils.h"
#include "utils/log_utils.h"
#include "utils/math_utils.h"
#include "utils/memory_utils.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace dorado::basecall {

namespace {

constexpr int MTL_CORE_BATCH_SIZE = 48;

}  // namespace

bool mps_flstm_is_broken() {
    static bool is_working = [] {
        // I don't know exactly when it was fixed, but it's broken in 14.7 and works in 15.5.
        // Note: I'm assuming that this is an OS issue and not a hardware one (ie it's not M2 vs M4)
        const NS::OperatingSystemVersion min_version{
                .majorVersion = 15,
                .minorVersion = 5,
                .patchVersion = 0,
        };
        const auto info = NS::ProcessInfo::processInfo();
        if (info->isOperatingSystemAtLeastVersion(min_version)) {
            return true;
        }

        DORADO_LOG(
                warn,
                "Torch's MPS backend is buggy for fLSTM before macOS {}.{}.{}. Falling back to CPU "
                "instead.",
                min_version.majorVersion, min_version.minorVersion, min_version.patchVersion);
        DORADO_LOG(
                warn,
                "Explicitly running with `--device cpu` is recommended on this device, as it will "
                "perform faster than allowing MPS fallback.");
        return false;
    }();
    return !is_working;
}

int mps_max_safe_batch_size(float memory_limit_fraction,
                            const config::BasecallModelConfig& model_config) {
    if (model_config.is_tx_model()) {
        // TODO: better number here
        return 32;
    }
    if (!model_config.is_flstm_model()) {
        throw std::runtime_error("Unknown model config");
    }
    if (mps_flstm_is_broken()) {
        // We're falling back on CPU, so limit it.
        // This matches the value in Models::set_basecaller_batch_params().
        return 128;
    }

    const int64_t physical_memory = utils::get_apple_physical_memory_bytes();
    const int64_t granularity = mps_batch_size_granularity(model_config);
    // We see performance issues from memory pressure, so only use part of what is available.
    const int64_t usable_memory = physical_memory * memory_limit_fraction / 2;

    // Memory usage is roughly given by the following:
    //   |memory_MB = 42 * batch_size + 1000|
    // Found empirically by running the following on an M3:
    //   |for bs in {64..512..32} ; do \time -l dorado basecaller -b ${bs} hac@v6 ... 2>&1 | grep "peak memory" ; done|
    int64_t batch_size = (usable_memory / 1024 / 1024 - 1000) / 42;
    batch_size = batch_size / granularity * granularity;

    if (batch_size <= 0) {
        DORADO_LOG(warn, "Failed to estimate max batch size: {}GB -> bs={}. Using {}",
                   usable_memory / utils::BYTES_PER_GB, batch_size, granularity);
        batch_size = granularity;
    }
    return static_cast<int>(batch_size);
}

int mps_batch_size_granularity(const config::BasecallModelConfig& model_config) {
    if (model_config.is_tx_model()) {
        return 8;
    }
    if (model_config.is_flstm_model()) {
        return 32;
    }
    throw std::runtime_error("Unknown model config");
}

int metal_lstm_max_safe_batch_size(float memory_limit_fraction,
                                   const config::BasecallModelConfig& model_config) {
    const size_t physical_memory = utils::get_apple_physical_memory_bytes();
    const size_t usable_memory = physical_memory * memory_limit_fraction;
    DORADO_LOG(debug, "Physical/Usable memory available: {}/{} GB",
               physical_memory / utils::BYTES_PER_GB, usable_memory / utils::BYTES_PER_GB);

    // Constrain decode buffers to about half of physical memory. Neural network GPU buffers and
    // CPU buffers are assumed to occupy a subset of the remainder.
    const auto out_chunk_size = model_config.basecaller.chunk_size() / model_config.stride;
    const auto states = decode::create_decoder_config(model_config).num_states();
    const auto decode_buffer_size_per_elem =
            static_cast<size_t>(out_chunk_size) *
            (static_cast<size_t>(model_config.outsize) +      // Scores
             static_cast<size_t>(states) * sizeof(int16_t) +  // Posts
             static_cast<size_t>(states) * sizeof(float));    // Back guides.
    DORADO_LOG(trace, "decode_buffer_size_per_elem {}", decode_buffer_size_per_elem);
    const int max_batch_size = static_cast<int>(std::clamp(
            utils::pad_to(usable_memory / (2 * decode_buffer_size_per_elem),
                          static_cast<size_t>(MTL_CORE_BATCH_SIZE)),
            static_cast<size_t>(MTL_CORE_BATCH_SIZE),
            static_cast<size_t>(MTL_CORE_BATCH_SIZE * utils::get_mtl_device_core_count())));
    DORADO_LOG(trace, "max_batch_size {}", max_batch_size);
    return max_batch_size;
}

int metal_lstm_batch_size_granularity() { return MTL_CORE_BATCH_SIZE; }

uint64_t metal_memory_limit(float memory_limit_fraction) {
    // This assumes that Dorado is the only significant memory user on the machine.
    return utils::get_apple_physical_memory_bytes() * memory_limit_fraction;
}

}  // namespace dorado::basecall
