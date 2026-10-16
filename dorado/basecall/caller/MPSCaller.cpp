#include "MPSCaller.h"

#include "MetalCallerTask.h"
#include "decode/Decoder.h"
#include "decode/MetalDecoder.h"
#include "model/Model.h"
#include "model/TxModel.h"
#include "torch_utils/metal_utils.h"
#include "utils/log_utils.h"
#include "utils/memory_utils.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

using namespace dorado::utils;

namespace dorado::basecall {

namespace {

bool is_mps_flstm_broken() {
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

// Types to use for different parts of the pipeline.
constexpr at::ScalarType scores_dtype = at::kHalf;
constexpr at::ScalarType posts_dtype = at::kFloat;
at::TensorOptions get_crf_options(const config::BasecallModelConfig &config) {
    if (config.is_flstm_model() && is_mps_flstm_broken()) {
        // MPS:Half and MPS:Float are broken, and CPU:Half is slower than CPU:Float.
        return at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
    } else {
        return at::TensorOptions().device(at::kMPS).dtype(at::kHalf);
    }
}

}  // namespace

MPSCaller::MPSCaller(const config::BasecallModelConfig &model_config) : MetalCaller(model_config) {
    ScopedAutoReleasePool autorelease_pool;

    // Our metal builds assume shared memory, so it's safe to check host.
    if (auto total_mem = utils::total_host_memory_GB(); total_mem < 16) {
        DORADO_LOG(warn,
                   "Less than 16GB of memory available: {}GB detected. "
                   "This is below minimum spec and may cause issues",
                   total_mem);
    }

    // Create MPS objects.
    {
        m_device = get_mtl_device();
        m_command_queue = NS::TransferPtr(m_device->newCommandQueue());
        m_decode_complete_event = NS::TransferPtr(m_device->newSharedEvent());
    }

    // Cache common options.
    {
        const auto decoder_options = decode::create_decoder_options(model_config);

        if (model_config.is_tx_model() &&
            decoder_options.blank_score != model_config.tx->crf.blank_score) {
            DORADO_LOG(warn, "Transformer model config does not have the expected blank score");
        }
    }

    // Set configuration sizes.
    {
        assert(model_config.has_normalised_basecaller_params());

        m_in_chunk_size = model_config.basecaller.chunk_size();
        // Chunk size after decimation via convolution stride.
        m_out_chunk_size = m_in_chunk_size / model_config.stride;
        m_batch_size = model_config.basecaller.batch_size();

        if (m_batch_size == 0) {
            if (model_config.is_tx_model()) {
                // Testing shows that a batch size of 16 is optimal in most cases, except
                // on low memory machines where 16 can perform a lot worse than 8.
                // TODO: replace with implementation of autobatchsize calculation
                if (utils::total_host_memory_GB() < 16) {
                    m_batch_size = 8;
                } else {
                    m_batch_size = 16;
                }
            } else if (model_config.is_flstm_model()) {
                // For now pick the largest batch size we can since speed seems to scale with it.
                // TODO: replace with implementation of autobatchsize calculation
                m_batch_size = get_max_safe_batch_size(1, model_config);
            }
        }
    }

    m_decoder = std::make_unique<MetalDecoder>(model_config, m_device.get(), m_out_chunk_size,
                                               m_batch_size, 1, scores_dtype, posts_dtype, 1.f,
                                               MetalDecoder::ScanMode::Float);

    // Load the model and get ready for calling.
    m_model = model::load_model(model_config, get_crf_options(model_config));
    start_threads();
}

MPSCaller::~MPSCaller() = default;

at::Tensor MPSCaller::create_input_tensor() const {
    // NCT
    return at::zeros({m_batch_size, m_config.num_features, m_in_chunk_size}, at::kHalf);
}

int MPSCaller::get_max_safe_batch_size(float memory_limit_fraction,
                                       const config::BasecallModelConfig &model_config) {
    if (model_config.is_tx_model()) {
        // TODO: better number here
        return 32;

    } else if (model_config.is_flstm_model()) {
        if (is_mps_flstm_broken()) {
            // We're falling back on CPU, so limit it.
            // Note: this matches the value in Models::set_basecaller_batch_params().
            return 128;
        }

        const int64_t physical_memory = get_apple_physical_memory_bytes();
        const int64_t granularity = get_batch_size_granularity(model_config);

        // We see performance issues from memory pressure, so only try and use part of what's avaiable.
        const int64_t usable_memory = physical_memory * memory_limit_fraction / 2;

        // Memory usage is roughly given by the following:
        //   |memory_MB = 42 * batch_size + 1000|
        // Found empirically by running the following on an M3:
        //   |for bs in {64..512..32} ; do \time -l dorado basecaller -b ${bs} hac@v6 ... 2>&1 | grep "peak memory" ; done|
        int64_t batch_size = (usable_memory / 1024 / 1024 - 1000) / 42;

        // Round down to a multiple of the granularity.
        batch_size = batch_size / granularity * granularity;

        if (batch_size <= 0) {
            DORADO_LOG(warn, "Failed to estimate max batch size: {}GB -> bs={}. Using {}",
                       usable_memory / BYTES_PER_GB, batch_size, granularity);
            batch_size = granularity;
        }
        return batch_size;

    } else {
        throw std::runtime_error("Unknown model config");
    }
}

int MPSCaller::get_batch_size_granularity(const config::BasecallModelConfig &model_config) {
    if (model_config.is_tx_model()) {
        return 8;
    } else if (model_config.is_flstm_model()) {
        return 32;
    } else {
        throw std::runtime_error("Unknown model config");
    }
}

bool MPSCaller::call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) {
    auto scores_TNC = m_model->forward(task.input->to(get_crf_options(m_config)))
                              .scores.transpose(0, 1)
                              .contiguous()
                              .to(scores_dtype);

    MTL::CommandBuffer *const cb = next_command_buffer(m_command_queue.get(), try_count);
    // wait for the previous decode task to complete - this acts as a mutex
    // previous scores are processed in the decode threads
    cb->encodeWait(m_decode_complete_event.get(), task.decode_complete_event_id - 1);

    m_decoder->scores().front().copy_(scores_TNC);

    std::lock_guard lock(inter_caller_mutex);
    return m_decoder->prepare_decode(cb, try_count);
}

DecodedData MPSCaller::decode(int chunk_idx) const { return m_decoder->decode(chunk_idx); };

}  // namespace dorado::basecall
