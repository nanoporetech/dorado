#include "MPSCaller.h"

#include "MetalCallerTask.h"
#include "basecall/crf_utils.h"
#include "decode/Decoder.h"
#include "decode/beam_search.h"
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
using torch::indexing::Slice;

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

CREATE_POINT_OF_INTEREST_ID(MPSCaller);

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

        m_bwd_scan_float_cps = make_cps(m_device.get(), "backward_scan_float", {}, std::nullopt);
        m_fwd_scan_add_softmax_float_cps =
                make_cps(m_device.get(), "forward_scan_add_softmax_float", {}, std::nullopt);
    }

    // Cache common options.
    {
        // TODO: non-Tx doesn't honour the config's n_base
        const int n_base = model_config.is_tx_model() ? model_config.tx->crf.n_base : 4;
        m_states = std::pow(n_base, model_config.state_len);

        m_decoder_options = decode::create_decoder_options(model_config);

        if (model_config.is_tx_model() &&
            m_decoder_options.blank_score != model_config.tx->crf.blank_score) {
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

    // Create tensors for scan/decode.
    {
        assert(m_out_chunk_size > 0);
        assert(m_batch_size > 0);
        assert(model_config.outsize > 0);

        const int T = m_out_chunk_size;
        const int C = model_config.outsize;
        const int Cs = m_states;
        const int N = m_batch_size;

        m_scores_TNC = torch::empty({T, N, C}, scores_dtype);
        m_posts_NTC = torch::empty({N, T + 1, Cs}, posts_dtype);
        m_bwd_NTC = torch::empty({N, T + 1, Cs});
    }

    // Load the model and get ready for calling.
    m_model = load_crf_model(model_config, get_crf_options(model_config));
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

bool MPSCaller::run_scan_kernels(MTL::CommandBuffer *const cb, int try_count) {
    POINT_OF_INTEREST_SCOPE(MPSCaller, run_scan_kernels, "try_count=%i", try_count);

    // ScanArgs expects scores TNC tensor sizes
    std::vector<int32_t> scan_args_{m_out_chunk_size, m_batch_size, m_states};
    auto scan_args = create_vec_buffer(m_device.get(), scan_args_);
    name_mtl_object(scan_args, "scan_kernel_args");

    // TODO: optimise grid size
    launch_kernel_no_wait(
            m_bwd_scan_float_cps.get(), cb,
            {scan_args.get(), mtl_for_tensor(m_scores_TNC), mtl_for_tensor(m_bwd_NTC)}, {},
            m_batch_size, m_states);

    launch_kernel_no_wait(m_fwd_scan_add_softmax_float_cps.get(), cb,
                          {scan_args.get(), mtl_for_tensor(m_scores_TNC), mtl_for_tensor(m_bwd_NTC),
                           mtl_for_tensor(m_posts_NTC)},
                          {}, m_batch_size, m_states);

    return run_command_buffer("linear/scan/softmax", cb, try_count);
}

bool MPSCaller::call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) {
    auto scores_TNC = m_model->forward(task.input->to(get_crf_options(m_config)))
                              .transpose(0, 1)
                              .contiguous()
                              .to(scores_dtype);

    MTL::CommandBuffer *const cb = next_command_buffer(m_command_queue.get(), try_count);
    // wait for the previous decode task to complete - this acts as a mutex
    // previous scores are processed in the decode threads
    cb->encodeWait(m_decode_complete_event.get(), task.decode_complete_event_id - 1);

    m_scores_TNC.index_put_({at::indexing::Ellipsis}, scores_TNC);

    std::lock_guard lock(inter_caller_mutex);
    return run_scan_kernels(cb, try_count);
}

DecodedData MPSCaller::decode(int chunk_idx) const {
    POINT_OF_INTEREST_SCOPE(MPSCaller, decode, "chunk_idx=%i", chunk_idx);

    // Not splitting batches in Tx impl so chunk idx should be in [0, N)
    assert(chunk_idx < m_batch_size);
    return decode::beam_search_decode(
            m_scores_TNC.index({Slice(), chunk_idx}), m_bwd_NTC[chunk_idx], m_posts_NTC[chunk_idx],
            m_decoder_options.beam_width, m_decoder_options.beam_cut, m_decoder_options.blank_score,
            m_decoder_options.q_shift, m_decoder_options.q_scale, 1.0f);
};

}  // namespace dorado::basecall
