#include "CudaModel.h"

#include "ChunkBenchmarks.h"
#include "config/ModBaseModelConfig.h"  // IWYU pragma: keep for ModBaseModelConfig types
#include "decode/Decoder.h"
#include "model/Model.h"
#include "torch_utils/cuda_utils.h"
#include "utils/log_utils.h"
#include "utils/math_utils.h"
#include "utils/stats.h"

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <utility>

namespace dorado::basecall {

namespace {

constexpr float GB = 1.0e9f;

constexpr auto default_beam_width = decode::DecoderOptions{}.beam_width;

constexpr int NUM_KOI_THREADS = 6;

c10::cuda::CUDAStream get_stream_for_device(c10::Device device) {
    c10::cuda::CUDAGuard device_guard(device);
    return c10::cuda::getStreamFromPool(false, device.index());
}

std::unique_ptr<nn::AuxiliaryData> create_empty_input(at::Tensor &in,
                                                      const at::TensorOptions &in_options,
                                                      at::Tensor &workspace,
                                                      const std::int32_t N,
                                                      const std::int32_t T,
                                                      const config::BasecallModelConfig &config,
                                                      nn::KoiThreads &thread_pool) {
    const std::int32_t C = config.num_features;
    const std::int32_t stride_out = config.stride;
    const std::int32_t stride_in = config.stride_inner();
    const std::int32_t chunk_size_granularity = config.chunk_size_granularity();
    const std::int32_t max_chunk_size = config.basecaller.chunk_size();
    const bool is_tx_model = config.is_tx_model();
    if (is_tx_model) {
        int max_conv_padding = 0;
        for (const auto &conv : config.convs) {
            max_conv_padding = std::max(max_conv_padding, conv.winlen / 2);
        }
        in = torch::empty({1, C, (N * T) + ((N + 1) * max_conv_padding)}, in_options);
    } else {
        in = torch::empty({1, C, N * T}, in_options);
        auto workspace_options =
                at::TensorOptions().device(torch::kCPU).pinned_memory(true).dtype(torch::kInt32);
        // for workspace size see koi/utils_lstm.h
        workspace = torch::empty({6 * ((T / stride_out) + 3) * N}, workspace_options);
    }

    auto aux = std::make_unique<nn::AuxiliaryData>(
            workspace, N, T, stride_out, stride_in, chunk_size_granularity,
            std::vector<std::int32_t>(N, T), max_chunk_size, is_tx_model);
    aux->create_auxiliary_data(in_options.device(), thread_pool);
    return aux;
}

}  // namespace

CudaRunnerState::CudaRunnerState(const CudaModel &model, size_t batch_dims_idx)
        : m_batch_size(model.batch_size(batch_dims_idx)),
          m_chunk_size(model.chunk_size(batch_dims_idx)),
          m_variable_chunk_sizes(model.variable_chunk_sizes()),
          m_is_tx_model(model.config().is_tx_model()),
          m_stride(model.config().stride),
          m_stride_inner(model.config().stride_inner()),
          m_chunk_size_granularity(model.config().chunk_size_granularity()),
          m_max_chunk_size(model.config().basecaller.chunk_size()),
          m_stream(c10::cuda::getStreamFromPool(false, model.device().index())) {
    std::tie(m_input, m_output, m_aux) = model.create_input_output_tensor(batch_dims_idx);
    if (m_is_tx_model) {
        m_first_conv_padding = model.config().convs.front().winlen / 2;
        m_chunk_offset = m_first_conv_padding;
        m_input.narrow(-1, 0, m_first_conv_padding).zero_();
    }
}

void CudaRunnerState::accept_chunk(int chunk_idx, const at::Tensor &chunk) {
    if (!m_variable_chunk_sizes) {
        m_input.index_put_({chunk_idx}, chunk);
        return;
    }

    assert(m_input.dim() == 3);
    assert(m_input.size(0) == 1);
    if ((m_input.size(1) == 1) && (m_input.dtype() == chunk.dtype())) {
        assert(chunk.dim() == 2);
        assert(chunk.size(0) == 1);
        std::memcpy(static_cast<std::byte *>(m_input.data_ptr()) +
                            (m_chunk_offset * chunk.element_size()),
                    chunk.data_ptr(), chunk.size(1) * chunk.element_size());
    } else {
        m_input.narrow(-1, m_chunk_offset, chunk.size(1)).copy_(chunk);
    }
    m_chunk_sizes.emplace_back(chunk.size(1));
    m_chunk_offset += m_chunk_sizes.back();
    if (m_is_tx_model) {
        m_input.narrow(-1, m_chunk_offset, m_first_conv_padding).zero_();
        m_chunk_offset += m_first_conv_padding;
    }
}

void CudaRunnerState::reset() {
    if (m_variable_chunk_sizes) {
        m_chunk_sizes.clear();
        m_input.narrow(-1, 0, m_first_conv_padding).zero_();
        m_chunk_offset = m_first_conv_padding;
    }
}

std::unique_ptr<nn::AuxiliaryData> CudaRunnerState::create_auxiliary_data() const {
    if (!m_variable_chunk_sizes) {
        return nullptr;
    }
    return std::make_unique<nn::AuxiliaryData>(m_aux, m_batch_size, m_chunk_size, m_stride,
                                               m_stride_inner, m_chunk_size_granularity,
                                               m_chunk_sizes, m_max_chunk_size, m_is_tx_model);
}

CudaModel::CudaModel(const BasecallerCreationParams &params)
        : m_config(params.model_config),
          m_device(params.device),
          m_decoder(decode::create_decoder(params.device, m_config, params.modbase_configs)),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(params.device)),
          m_low_latency(params.pipeline_type == PipelineType::simplex_low_latency),
          m_pipeline_type(params.pipeline_type),
          m_stream(get_stream_for_device(m_options.device())),
          m_variable_chunk_sizes(params.variable_chunk_sizes),
          m_thread_pool(NUM_KOI_THREADS) {
    assert(m_options.device().is_cuda());
    assert(params.model_config.has_normalised_basecaller_params());

    m_decoder_options = decode::create_decoder_options(params.model_config);
    m_num_input_features = params.model_config.num_features;

    // If we allow this to be changed then cuda_memory_requirements() will
    // need updating.
    if (m_decoder_options.beam_width != default_beam_width) {
        throw std::logic_error("Decoder beam width is no longer constant");
    }

    // Anything past this point should use the requested device.
    c10::cuda::CUDAGuard device_guard(m_options.device());
    c10::cuda::CUDACachingAllocator::emptyCache();

    at::InferenceMode guard;

    if (!std::empty(params.modbase_configs)) {
        if (!m_config.is_flstm_model()) {
            throw std::runtime_error(
                    "Latent-space modbase models require an FLSTM basecalling model");
        }
        if (m_variable_chunk_sizes) {
            throw std::runtime_error(
                    "Variable chunk sizes are not currently supported with latent space modbase "
                    "models.");
        }
        // TODO: check the required dtype here
        m_modbase_modules = modbase::load_modbase_latent_space_model(
                params.modbase_configs, m_options,
                static_cast<int>(m_config.basecaller.chunk_size() / m_config.stride_inner()));

        m_modbase_info = modbase::create_modbase_info(params.modbase_configs);
    }
    m_model = model::load_model(params.model_config, m_options);

    determine_batch_dims(params);

    auto [crfmodel_bytes_per_ct, decode_bytes_per_ct] = cuda::memory_requirements(m_config);

    // Warmup
    c10::cuda::CUDAStreamGuard stream_guard(m_stream);
    for (const auto &batch_dim : m_batch_dims) {
        DORADO_LOG(debug, "{} using chunk size {}, batch size {}", m_device, batch_dim.T_in,
                   batch_dim.N);
        DORADO_LOG(debug, "{} Model memory {:.2f}GB", m_device,
                   (crfmodel_bytes_per_ct * batch_dim.T_out * batch_dim.N) / GB);
        DORADO_LOG(debug, "{} Decode memory {:.2f}GB", m_device,
                   (decode_bytes_per_ct * batch_dim.T_out * batch_dim.N) / GB);
        at::Tensor input;
        at::Tensor workspace;
        std::unique_ptr<nn::AuxiliaryData> aux;
        if (m_variable_chunk_sizes) {
            aux = create_empty_input(input, m_options, workspace, batch_dim.N, batch_dim.T_in,
                                     m_config, m_thread_pool);
        } else {
            input = torch::empty({batch_dim.N, m_num_input_features, batch_dim.T_in}, m_options);
        }
        auto model_output = m_model->forward(input, aux.get(), m_modbase_modules.has_value());
        auto decoded_data = m_decoder->prepare_decode(
                {model_output.scores, batch_dim.N, m_decoder_options, aux.get()});
        if (m_modbase_modules) {
            modbase::run_modbase_latent_space_models(*m_modbase_modules, decoded_data.data,
                                                     std::move(model_output.signal_embeddings));
        }
    }
    m_stream.synchronize();

    // Callers run on a different thread, so clean up any unreferenced allocations from this one
    // otherwise they'll stick around forever.
    c10::cuda::CUDACachingAllocator::emptyCache();
}

CudaModel::~CudaModel() = default;

std::unique_ptr<RunnerState> CudaModel::create_runner_state(size_t batch_dims_idx) {
    return std::make_unique<CudaRunnerState>(*this, batch_dims_idx);
}

// If 5 minutes has passed since the first chunk was added to a batch, we will
// dispatch the batch even if it is not full. This is to prevent issues with
// MinKNOW clients disconnecting because they think the server has timed out.
static constexpr int DEFAULT_FIRST_CHUNK_TIMEOUT_MS = 300000;

// If 30 seconds has passed since the most recent chunk was added to a batch,
// we will dispatch the batch even if it is not full. Benchmarking indicates
// that this gives good results when some pipelines have very low throughput
// and others have very high throughput.
static constexpr int DEFAULT_LAST_CHUNK_TIMEOUT_MS = 30000;

// Default value for timeout of incomplete batches for low-latency pipelines. The
// value of 350 ms has been found to give good adaptive-sampling performance on all
// platforms. For low-latency pipelines the timeout is always from when the first
// chunk was added to the batch.
static constexpr int DEFAULT_LOW_LATENCY_TIMEOUT_MS = 350;

std::pair<int, int> CudaModel::batch_timeouts_ms() const {
    // For low-latency pipelines we set both timeouts to the same value. This means that we
    // will always timeout based on the time from the first chunk being added to the batch.
    return m_low_latency
                   ? std::make_pair(DEFAULT_LOW_LATENCY_TIMEOUT_MS, DEFAULT_LOW_LATENCY_TIMEOUT_MS)
                   : std::make_pair(DEFAULT_FIRST_CHUNK_TIMEOUT_MS, DEFAULT_LAST_CHUNK_TIMEOUT_MS);
}

SchedulingPolicy CudaModel::scheduling_policy() const {
    // Global task queues, one per GPU. This ensures that tasks from different clients are
    // processed in the order they became ready, rather than individual callers contending
    // for a global mutex that can wake up threads in an arbitrary order.
    SchedulingPolicy policy;
    policy.shared_queue_key =
            "cuda:" + std::to_string(m_options.device().index()) + (m_low_latency ? ":ll" : "");
    // Low-latency callers are released as soon as their batch is dispatched, so that any other
    // low-latency callers that are ready can get going immediately. Otherwise callers using the
    // same GPU must not start a new batch before this one has completed.
    policy.release_point =
            m_low_latency ? ReleasePoint::on_submit_start : ReleasePoint::on_submit_return;
    return policy;
}

namespace {

class CudaThreadScope final : public ThreadScope {
public:
    explicit CudaThreadScope(const c10::cuda::CUDAStream &stream) : m_stream_guard(stream) {}

private:
    c10::cuda::CUDAStreamGuard m_stream_guard;
};

struct CudaBatch final : InFlightBatch {
    CudaBatch(CudaRunnerState &state_,
              std::unique_ptr<nn::AuxiliaryData> aux_,
              at::Tensor input_,
              int num_chunks_)
            : state(state_),
              aux(std::move(aux_)),
              input(std::move(input_)),
              num_chunks(num_chunks_) {}
    CudaRunnerState &state;
    std::unique_ptr<nn::AuxiliaryData> aux;
    at::Tensor input;  // On the device.
    int num_chunks;
    decode::DecoderResult out;
};

}  // namespace

std::unique_ptr<ThreadScope> CudaModel::worker_thread_scope() {
    return std::make_unique<CudaThreadScope>(m_stream);
}

std::unique_ptr<InFlightBatch> CudaModel::prepare(RunnerState &runner_state, int num_chunks) {
    NVTX3_FUNC_RANGE();
    auto &state = static_cast<CudaRunnerState &>(runner_state);
    c10::cuda::CUDAStreamGuard guard(state.m_stream);

    auto aux = state.create_auxiliary_data();
    if (!aux && m_variable_chunk_sizes) {
        throw std::logic_error("Missing auxiliary data while calling variable chunks!");
    }
    if (aux && !m_variable_chunk_sizes) {
        throw std::logic_error("Found auxiliary data while calling fixed chunks!");
    }

    at::Tensor device_input = state.m_input.to(m_options.device());  // async copy

    if (aux) {
        aux->create_auxiliary_data(m_options.device(), m_thread_pool);
    }
    return std::make_unique<CudaBatch>(state, std::move(aux), std::move(device_input), num_chunks);
}

void CudaModel::submit(InFlightBatch &in_flight) {
    NVTX3_FUNC_RANGE();
    auto &batch = static_cast<CudaBatch &>(in_flight);
    auto device_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(m_options.device().index());

    const auto aggregate_idx = static_cast<uint64_t>(c10::CachingAllocator::StatType::AGGREGATE);
    DORADO_LOG(trace,
               "Aggregate current: allocation {}, segment {}, active {}, inactive_split {}, "
               "alloc_bytes {}, reserved_bytes {}, active_bytes {}, inactive_split_bytes {}, "
               "requested_bytes {}. Total: num_alloc_retries {}, num_ooms {}, max_split_size {}",
               device_stats.allocation[aggregate_idx].current,
               device_stats.segment[aggregate_idx].current,
               device_stats.active[aggregate_idx].current,
               device_stats.inactive_split[aggregate_idx].current,
               device_stats.allocated_bytes[aggregate_idx].current,
               device_stats.reserved_bytes[aggregate_idx].current,
               device_stats.active_bytes[aggregate_idx].current,
               device_stats.inactive_split_bytes[aggregate_idx].current,
               device_stats.requested_bytes[aggregate_idx].current, device_stats.num_alloc_retries,
               device_stats.num_alloc_retries, device_stats.num_ooms, device_stats.max_split_size);

    auto run_basecalling = [&]() {
        stats::Timer timer;
        auto model_output =
                m_model->forward(batch.input, batch.aux.get(), m_modbase_modules.has_value());
        batch.out = m_decoder->prepare_decode({std::move(model_output.scores), batch.num_chunks,
                                               m_decoder_options, batch.aux.get()});
        if (m_modbase_modules) {
            batch.out.modbase_data = modbase::run_modbase_latent_space_models(
                    *m_modbase_modules, batch.out.data, std::move(model_output.signal_embeddings));
        }
        m_stream.synchronize();
        m_model_decode_ms += timer.GetElapsedMS();
    };

    try {
        run_basecalling();
    } catch (c10::Error &e) {
        DORADO_LOG(warn, "Caught Torch error '{}', clearing CUDA cache and retrying.", e.msg());
        c10::cuda::CUDACachingAllocator::emptyCache();
        run_basecalling();
    }
    ++m_num_batches_called;
}

void CudaModel::decode(InFlightBatch &in_flight,
                       size_t /*unit*/,
                       std::vector<decode::DecodedChunk> &out) {
    auto &batch = static_cast<CudaBatch &>(in_flight);
    auto &state = batch.state;
    c10::cuda::CUDAStreamGuard guard(state.m_stream);

    if (batch.aux && m_variable_chunk_sizes) {
        at::Tensor decoded =
                state.m_output.narrow(0, 0, c10::multiply_integers(batch.out.data.sizes()))
                        .view(batch.out.data.sizes());
        decoded.copy_(batch.out.data);
        out = m_decoder->decode(
                {decoded, {}, batch.num_chunks, m_decoder_options, batch.aux.get()});
        return;
    }

    state.m_output.copy_(batch.out.data);
    std::vector<at::Tensor> modbase_output;
    modbase_output.reserve(batch.out.modbase_data.size());
    for (const at::Tensor &head : batch.out.modbase_data) {
        // ToDo: use pinned memory
        modbase_output.emplace_back(head.to(at::kCPU));
    }
    out = m_decoder->decode(
            {state.m_output, std::move(modbase_output), batch.num_chunks, m_decoder_options});
}

void CudaModel::release(InFlightBatch &in_flight, std::vector<decode::DecodedChunk> & /*out*/) {
    static_cast<CudaBatch &>(in_flight).state.reset();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> CudaModel::create_input_output_tensor(
        size_t batch_dims_idx) const {
    auto opts = at::TensorOptions().device(torch::kCPU).pinned_memory(true);
    int64_t N = m_batch_dims[batch_dims_idx].N;
    int64_t T_in = m_batch_dims[batch_dims_idx].T_in;
    int64_t T_out = m_batch_dims[batch_dims_idx].T_out;
    int64_t C_in = m_num_input_features;
    auto scalar_type = c10::typeMetaToScalarType(m_options.dtype());
    // A runner's input and output buffers are never in use simultaneously, thus they can be mapped
    // to the same backing tensor.
    int64_t input_bytes = N * C_in * T_in * m_options.dtype().itemsize();
    int64_t output_bytes = 3 * N * T_out;
    auto storage = torch::empty({std::max(input_bytes, output_bytes)}, opts.dtype(torch::kInt8));
    if (m_variable_chunk_sizes) {
        auto input = storage.slice(0, 0, input_bytes).view(scalar_type).view({1, C_in, N * T_in});
        std::int64_t aux_size = 0;
        if (m_config.is_lstm_model() || m_config.is_flstm_model()) {
            // for workspace size see koi/utils_lstm.h
            aux_size = 6 * (T_out + 3) * N;
        }
        at::Tensor output = storage.slice(0, 0, output_bytes);
        at::Tensor aux = torch::empty({aux_size}, opts.dtype(torch::kInt32));
        return {input, output, aux};
    }
    auto input = storage.slice(0, 0, input_bytes).view(scalar_type).view({N, C_in, T_in});
    auto output = storage.slice(0, 0, output_bytes).view({3, N, T_out});
    return {input, output, at::Tensor{}};
}

stats::NamedStats CudaModel::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = static_cast<double>(m_num_batches_called);
    stats["model_decode_ms"] = static_cast<double>(m_model_decode_ms);
    return stats;
}

void CudaModel::determine_batch_dims(const BasecallerCreationParams &params) {
    const int requested_batch_size = m_config.basecaller.batch_size();
    auto [batch_dims, max_batch_sizes] = cuda::calculate_batch_sizes(
            m_options.device(), params.memory_limit_fraction, m_config, m_pipeline_type,
            requested_batch_size, m_variable_chunk_sizes);
    m_batch_dims = std::move(batch_dims);

    if (requested_batch_size != 0 || max_batch_sizes.empty()) {
        return;
    }

    assert(m_batch_dims.size() == max_batch_sizes.size());

    // We limit the maximum when doing benchmarking to avoid excessive startup time.
    // The limit for transformer models should be increased at a later time.
    int max_batch_size = *std::max_element(max_batch_sizes.begin(), max_batch_sizes.end());
    const int max_batch_size_limit = m_config.is_tx_model() ? 1024 : 10240;
    max_batch_size = std::min(max_batch_size, max_batch_size_limit);

    const int chunk_granularity = m_config.chunk_size_granularity();
    const int batch_granularity = cuda::batch_size_granularity(m_config);
    const int stride = m_config.stride;

    // `288 * stride` (much shorter than the default chunk size of 10k), adjusted for
    // granularity, is a somewhat arbitrary trade-off between getting accurate measurements
    // and avoiding excessive startup time
    const int chunk_size = utils::pad_to(288 * stride, chunk_granularity);
    DORADO_LOG(debug, "Auto batchsize {}: testing up to {} in steps of {}", m_device,
               max_batch_size, batch_granularity);

    // Times and corresponding batch sizes.
    std::vector<std::pair<float, int>> times_and_batch_sizes;
    times_and_batch_sizes.reserve(max_batch_size / batch_granularity);

    const std::string model_name = m_config.model_path.filename().string();

    // See if we can find cached values for the chunk timings for this run condition
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    const auto chunk_benchmarks =
            ChunkBenchmarks::instance().get_chunk_timings(prop->name, model_name);
    if (!chunk_benchmarks) {
        DORADO_LOG(
                info,
                "Calculating optimized batch size for GPU \"{}\" and model {}. Full benchmarking "
                "will run for this device, which may take some time. Consider using "
                "--run-batchsize-benchmarks to save these benchmarks to disk, as they can be "
                "used in future runs with --batchsize-benchmarks-file.",
                prop->name, model_name);
    }

    float best_time = std::numeric_limits<float>::max();
    for (int batch_size = batch_granularity; batch_size <= max_batch_size;
         batch_size += batch_granularity) {
        float time = std::numeric_limits<float>::max();

        // Use the available cached chunk size if we haven't been explicitly told not to.
        if (chunk_benchmarks) {
            // Note that if a cache of batch size timings is available, we don't mix cached and live
            //  benchmarks, to avoid discontinuities in the data.
            if (chunk_benchmarks->find(batch_size) != chunk_benchmarks->end()) {
                time = chunk_benchmarks->at(batch_size);
            }
        } else {
            at::Tensor input;
            at::Tensor workspace;
            std::unique_ptr<nn::AuxiliaryData> aux;
            if (m_variable_chunk_sizes) {
                aux = create_empty_input(input, m_options, workspace, batch_size, chunk_size,
                                         m_config, m_thread_pool);
            } else {
                input = torch::empty({batch_size, m_config.num_features, chunk_size}, m_options);
            }

            for (int i = 0; i < 2; ++i) {  // run twice to eliminate outliers
                if (m_variable_chunk_sizes) {
                    // Need to reset this for the second run as TxModel may update it
                    aux->set_chunk_size_granularity(m_config.chunk_size_granularity());
                }
                using utils::handle_cuda_result;
                cudaEvent_t start, stop;
                handle_cuda_result(cudaEventCreate(&start));
                handle_cuda_result(cudaEventCreate(&stop));
                handle_cuda_result(cudaEventRecord(start));
                m_model->forward(input, aux.get());
                handle_cuda_result(cudaEventRecord(stop));
                handle_cuda_result(cudaEventSynchronize(stop));
                float ms = 0;
                handle_cuda_result(cudaEventElapsedTime(&ms, start, stop));
                auto time_this_iteration = ms / batch_size;
                time = std::min(time, time_this_iteration);
                handle_cuda_result(cudaEventDestroy(start));
                handle_cuda_result(cudaEventDestroy(stop));
                if (aux && aux->is_lstm_or_flstm_model()) {
                    aux->restore_convolution_auxiliary_data();
                }
                DORADO_LOG(trace, "Auto batchsize {}: iteration:{}, ms/chunk {:8f} ms", m_device, i,
                           time_this_iteration);
            }
            // Clear the cache each time. Without this, intermittent cuda memory allocation errors
            // are seen on windows laptop NVIDIA RTX A5500 Laptop GPU. See JIRA issue DOR-466
            c10::cuda::CUDACachingAllocator::emptyCache();

            DORADO_LOG(debug, "Auto batchsize {}: {}, time per chunk {:8f} ms", m_device,
                       batch_size, time);
        }

        if (time < best_time) {
            best_time = time;
            times_and_batch_sizes.emplace_back(time, batch_size);
        }
    }

    if (!chunk_benchmarks) {
        // If we have just generated benchmarks that didn't previously exist, add them to the in-memory cache. This
        // will be of benefit to basecall servers which won't have to keep re-generating the benchmarks each time a
        // runner is created.
        ChunkBenchmarks::instance().add_chunk_timings(prop->name, model_name,
                                                      times_and_batch_sizes);

        DORADO_LOG(debug,
                   "Adding chunk timings to internal cache for GPU {}, model {} ({} "
                   "entries)",
                   prop->name, model_name, times_and_batch_sizes.size());
    }

    // Find the first batch size that was under the threshold.
    const float threshold_time = best_time * (1 + params.batch_size_time_penalty);
    auto under_threshold = [threshold_time](auto pair) { return pair.first <= threshold_time; };
    auto largest_usable_batch = std::find_if(times_and_batch_sizes.begin(),
                                             times_and_batch_sizes.end(), under_threshold);
    if (largest_usable_batch == times_and_batch_sizes.end()) {
        // This should be impossible.
        // Sanity check only, to avoid segfault or misleading behavior if there is a bug.
        throw std::out_of_range("Error in batch size selection algorithm.");
    }
    DORADO_LOG(debug, "Largest batch size for {}: {}, time per chunk {:8f} ms", m_device,
               largest_usable_batch->second, largest_usable_batch->first);

    for (size_t i = 0; i < m_batch_dims.size(); ++i) {
        // Pick the largest batch size under the max.
        int &final_size = m_batch_dims[i].N;
        const int max_size = max_batch_sizes[i];
        for (auto it = times_and_batch_sizes.begin(); it != std::next(largest_usable_batch); ++it) {
            const int batch_size = it->second;
            if (batch_size <= max_size) {
                final_size = batch_size;
            }
        }
        DORADO_LOG(debug, "Final batch size for {}[{}]: {}", m_device, i, final_size);
    }
}

}  // namespace dorado::basecall
