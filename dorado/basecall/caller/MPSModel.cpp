#include "MPSModel.h"

#include "MetalCallerPlanning.h"
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

// Types to use for different parts of the pipeline.
constexpr at::ScalarType scores_dtype = at::kHalf;
constexpr at::ScalarType posts_dtype = at::kFloat;
at::TensorOptions get_crf_options(const config::BasecallModelConfig &config) {
    if (config.is_flstm_model() && mps::flstm_is_broken()) {
        // MPS:Half and MPS:Float are broken, and CPU:Half is slower than CPU:Float.
        return at::TensorOptions().device(at::kCPU).dtype(at::kFloat);
    } else {
        return at::TensorOptions().device(at::kMPS).dtype(at::kHalf);
    }
}

}  // namespace

MPSModel::MPSModel(const config::BasecallModelConfig &model_config,
                   const std::span<const config::ModBaseModelConfig> modbase_configs)
        : MetalModel(model_config), m_modbase_decoder(modbase_configs) {
    ScopedAutoReleasePool autorelease_pool;

    // Create MPS objects.
    {
        m_command_queue = NS::TransferPtr(m_device->newCommandQueue());
    }

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
                m_batch_size = mps::max_safe_batch_size(1, model_config);
            }
        }
    }

    m_decoder = std::make_unique<MetalDecoder>(model_config, m_device.get(), m_out_chunk_size,
                                               m_batch_size, 1, scores_dtype, posts_dtype, 1.f,
                                               MetalDecoder::ScanMode::Float);

    // Load the model and get ready for calling.
    m_model = model::load_model(model_config, get_crf_options(model_config));

    if (!modbase_configs.empty()) {
        if (!model_config.is_flstm_model()) {
            throw std::runtime_error(
                    "Latent-space modbase models require an FLSTM basecalling model");
        }
        m_modbase_modules = modbase::load_modbase_latent_space_model(
                modbase_configs, get_crf_options(model_config),
                static_cast<int>(model_config.basecaller.chunk_size() /
                                 model_config.stride_inner()));
        m_modbase_info = modbase::create_modbase_info(modbase_configs);
    }
}

MPSModel::~MPSModel() = default;

at::Tensor MPSModel::create_input_tensor() const {
    // NCT
    return at::zeros({m_batch_size, m_config.num_features, m_in_chunk_size}, at::kHalf);
}

bool MPSModel::call_task(MetalCallerTask &task, std::mutex &inter_caller_mutex, int try_count) {
    auto model_output = m_model->forward(task.input->to(get_crf_options(m_config)), nullptr,
                                         m_modbase_modules.has_value());
    auto scores_TNC = model_output.scores.transpose(0, 1).contiguous().to(scores_dtype);
    task.signal_embeddings = std::move(model_output.signal_embeddings);

    m_decoder->scores().front().copy_(scores_TNC);

    // wait for the previous decode task to complete - this acts as a mutex
    // previous scores are processed in the decode threads
    MTL::CommandBuffer *const cb = next_command_buffer_after_event(
            m_command_queue.get(), try_count, decode_complete_event(),
            task.decode_complete_event_id - 1);

    std::lock_guard lock(inter_caller_mutex);
    return m_decoder->prepare_decode(cb, try_count);
}

void MPSModel::decode_modbase(MetalCallerTask &task, std::span<decode::DecodedChunk> chunks) {
    if (!m_modbase_modules) {
        return;
    }

    {
        std::lock_guard lock(m_modbase_mutex);
        auto heads = modbase::run_modbase_latent_space_models(
                *m_modbase_modules,
                decode::pack_decoded_chunks(chunks).to(task.signal_embeddings.device()),
                task.signal_embeddings.narrow(0, 0, task.num_chunks));
        for (at::Tensor &head : heads) {
            head = head.to(at::kCPU);
        }
        m_modbase_decoder.decode(heads, chunks);
    }
    task.signal_embeddings = at::Tensor{};
}

}  // namespace dorado::basecall
