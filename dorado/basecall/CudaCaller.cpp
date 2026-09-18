#include "basecall/CudaCaller.h"

#include "ChunkBenchmarks.h"
#include "basecall/crf_utils.h"
#include "config/ModBaseModelConfig.h"  // IWYU pragma: keep for ModBaseModelConfig types
#include "decode/Decoder.h"
#include "torch_utils/cuda_utils.h"
#include "utils/math_utils.h"
#include "utils/thread_utils.h"

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <set>

namespace dorado::basecall {

namespace {

struct NNTask {
    NNTask(at::Tensor input_, int num_chunks_, nn::AuxiliaryData *aux_, void *caller_)
            : input(std::move(input_)), num_chunks(num_chunks_), aux(aux_), caller(caller_) {}
    at::Tensor input;
    int num_chunks;
    nn::AuxiliaryData *aux;
    void *caller;
    decode::DecodeData out;
    std::mutex mut;
    std::condition_variable cv;
    bool done{false};
};

constexpr float GB = 1.0e9f;

constexpr auto default_beam_width = decode::DecoderOptions{}.beam_width;

c10::cuda::CUDAStream get_stream_for_device(c10::Device device) {
    c10::cuda::CUDAGuard device_guard(device);
    return c10::cuda::getStreamFromPool(false, device.index());
}

void run_modbase_models(modbase::ModBaseLatentSpaceModels &models,
                        const at::Tensor &decoded_data,
                        const at::Tensor &signal_embeddings) {
    auto &[sequence_model, common_model, classifier_models] = models;
    auto encoded_sequence = sequence_model->forward(decoded_data);
    auto common_output = common_model->forward(encoded_sequence, signal_embeddings);
    for (auto &classifier_model : classifier_models) {
        classifier_model->forward(common_output);
    }
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

static constexpr int NUM_KOI_THREADS = 6;

struct CudaCaller::GPUTaskQueue {
public:
    std::queue<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
};

struct CudaCaller::BatchDimsAndMaxSizes {
    std::vector<CudaCaller::BatchDims> batch_dims;
    // |max_batch_sizes| will either be the same size as |batch_dims|, or empty
    // if an error occurred.
    std::vector<int> max_batch_sizes;
};

CudaCaller::CudaCaller(const BasecallerCreationParams &params)
        : m_config(params.model_config),
          m_device(params.device),
          m_decoder(decode::create_decoder(params.device, m_config)),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(params.device)),
          m_low_latency(params.pipeline_type == PipelineType::simplex_low_latency),
          m_pipeline_type(params.pipeline_type),
          m_stream(get_stream_for_device(m_options.device())),
          m_variable_chunk_sizes(params.variable_chunk_sizes),
          m_thread_pool(NUM_KOI_THREADS) {
    assert(m_options.device().is_cuda());
    assert(params.model_config.has_normalised_basecaller_params());

    m_decoder_options.q_shift = params.model_config.qbias;
    m_decoder_options.q_scale = params.model_config.qscale;
    m_num_input_features = params.model_config.num_features;

    // If we allow this to be changed then calculate_memory_requirements() will
    // need updating.
    if (m_decoder_options.beam_width != default_beam_width) {
        throw std::logic_error("Decoder beam width is no longer constant");
    }

    // Anything past this point should use the requested device.
    c10::cuda::CUDAGuard device_guard(m_options.device());
    c10::cuda::CUDACachingAllocator::emptyCache();

    at::InferenceMode guard;

    if (params.modbase_configs && !params.modbase_configs->empty()) {
        if (!m_config.is_flstm_model()) {
            throw std::runtime_error(
                    "Latent-space modbase models require an FLSTM basecalling model");
        }
        // TODO: check the required dtype here
        m_modbase_modules =
                modbase::load_modbase_latent_space_model(*(params.modbase_configs), m_options);
    }
    m_module = load_crf_model(params.model_config, m_options);

    determine_batch_dims(params, m_variable_chunk_sizes);

    auto [crfmodel_bytes_per_ct, decode_bytes_per_ct] = calculate_memory_requirements(m_config);

    // Warmup
    c10::cuda::CUDAStreamGuard stream_guard(m_stream);
    for (const auto &batch_dim : m_batch_dims) {
        spdlog::debug("{} using chunk size {}, batch size {}", m_device, batch_dim.T_in,
                      batch_dim.N);
        spdlog::debug("{} Model memory {:.2f}GB", m_device,
                      (crfmodel_bytes_per_ct * batch_dim.T_out * batch_dim.N) / GB);
        spdlog::debug("{} Decode memory {:.2f}GB", m_device,
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
        at::Tensor signal_embeddings;
        auto scores = m_modbase_modules ? m_module->forward(input, aux.get(), &signal_embeddings)
                                        : m_module->forward(input, aux.get());
        auto decoded_data =
                m_decoder->beam_search_part_1({scores, batch_dim.N, m_decoder_options, aux.get()});
        if (m_modbase_modules) {
            run_modbase_models(*m_modbase_modules, decoded_data.data, signal_embeddings);
        }
    }
    m_stream.synchronize();

    start_threads();

    // This isn't the thread that we run on, so clean up any unreferenced allocations
    // otherwise they'll stick around forever.
    c10::cuda::CUDACachingAllocator::emptyCache();
}

CudaCaller::~CudaCaller() { terminate(); }

CudaCaller::GPUTaskQueue &CudaCaller::get_task_queue() {
    // Global task queues, one per GPU. This ensures that tasks from different clients are
    // processed in the order they became ready, rather than individual callers contending
    // for a global mutex that can wake up threads in an arbitrary order

    static std::vector<GPUTaskQueue> gpu_task_queues(torch::cuda::device_count());
    static std::vector<GPUTaskQueue> low_latency_gpu_task_queues(torch::cuda::device_count());

    auto &task_queues = m_low_latency ? low_latency_gpu_task_queues : gpu_task_queues;
    return task_queues.at(m_options.device().index());
}

std::pair<int, int> CudaCaller::batch_timeouts_ms() const {
    // For low-latency pipelines we set both timeouts to the same value. This means that we
    // will always timeout based on the time from the first chunk being added to the batch.
    return m_low_latency
                   ? std::make_pair(DEFAULT_LOW_LATENCY_TIMEOUT_MS, DEFAULT_LOW_LATENCY_TIMEOUT_MS)
                   : std::make_pair(DEFAULT_FIRST_CHUNK_TIMEOUT_MS, DEFAULT_LAST_CHUNK_TIMEOUT_MS);
}

int CudaCaller::get_batch_size_granularity(const config::BasecallModelConfig &model_config) {
    // TODO: we may want to use different numbers based on model type and GPU arch
    return model_config.is_tx_model() ? 32 : 64;
}

int64_t CudaCaller::get_gpu_mem_limit(c10::Device device, float memory_limit_fraction) {
    c10::cuda::CUDAGuard device_guard(device);
    c10::cuda::CUDACachingAllocator::emptyCache();
    const int64_t available = utils::available_memory(device);
    spdlog::debug("{} memory available: {:.2f}GB", device.str(), available / GB);

    // If running on a Jetson device with unified memory for CPU and GPU we can't use all
    // the available memory for GPU tasks. This way we leave at least half for the CPU,
    // though it's not clear what the ideal split would be.
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
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
        // restrict Orin and Thor further as there's no benefit to the largest batch sizes
        // and definite down sides to using all the memory
        memory_limit_fraction *= 0.5f;
    }

    // Apply limit fraction.
    int64_t gpu_mem_limit = static_cast<int64_t>(available * memory_limit_fraction);

    // Allow 1GB for model weights, etc.
    gpu_mem_limit -= GB;
    // Torch and CUDA internals reserve about another GB.
    gpu_mem_limit -= GB;

    if (gpu_mem_limit < 0) {
        throw std::runtime_error(
                "Failed to determine safe batch size. Less than 1GB GPU memory available.");
    }
    return gpu_mem_limit;
}

std::vector<decode::DecodedChunk> CudaCaller::call_chunks(at::Tensor &input,
                                                          at::Tensor &output,
                                                          int num_chunks,
                                                          nn::AuxiliaryData *const aux) {
    NVTX3_FUNC_RANGE();
    if (num_chunks == 0) {
        return std::vector<decode::DecodedChunk>();
    }

    if (!aux && m_variable_chunk_sizes) {
        throw std::logic_error("Missing auxiliary data while calling variable chunks!");
    }
    if (aux && !m_variable_chunk_sizes) {
        throw std::logic_error("Found auxiliary data while calling fixed chunks!");
    }

    at::Tensor device_input = input.to(m_options.device());  // async copy

    if (aux) {
        aux->create_auxiliary_data(m_options.device(), m_thread_pool);
    }

    auto &task_queue = get_task_queue();
    auto task = std::make_shared<NNTask>(device_input, num_chunks, aux, this);
    {
        std::lock_guard<std::mutex> lock(task_queue.m_input_lock);
        task_queue.m_input_queue.push(task);
    }
    task_queue.m_input_cv.notify_all();

    std::unique_lock lock(task->mut);
    while (!task->done) {
        task->cv.wait(lock);
    }

    if (aux && m_variable_chunk_sizes) {
        at::Tensor out = output.narrow(0, 0, c10::multiply_integers(task->out.data.sizes()))
                                 .view(task->out.data.sizes());
        out.copy_(task->out.data);
        return m_decoder->beam_search_part_2({out, num_chunks, m_decoder_options, task->aux});
    }

    output.copy_(task->out.data);
    return m_decoder->beam_search_part_2({output, num_chunks, m_decoder_options});
}

void CudaCaller::terminate() {
    m_terminate.store(true);
    auto &task_queue = get_task_queue();
    task_queue.m_input_cv.notify_all();
    if (m_cuda_thread.joinable()) {
        m_cuda_thread.join();
    }
}

void CudaCaller::restart() {
    // This can be called more than once, via multiple runners.
    if (m_terminate.exchange(false)) {
        start_threads();
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> CudaCaller::create_input_output_tensor(
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

stats::NamedStats CudaCaller::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = static_cast<double>(m_num_batches_called);
    stats["model_decode_ms"] = static_cast<double>(m_model_decode_ms);
    return stats;
}

int CudaCaller::get_max_safe_batch_size(c10::Device device,
                                        float memory_limit_fraction,
                                        const config::BasecallModelConfig &model_config) {
    const int requested_batch_size = 0;       // Determine limit for us.
    const auto pipeline_type = std::nullopt;  // Don't add extra chunk sizes.
    const bool variable_chunk_sizes = false;  // Doesn't matter when determining max_safe_batch_size
    auto max_batch_sizes =
            calculate_batch_sizes(device, memory_limit_fraction, model_config, pipeline_type,
                                  requested_batch_size, variable_chunk_sizes)
                    .max_batch_sizes;
    // We should only have the one result since we didn't request the extra chunk sizes.
    if (max_batch_sizes.size() != 1) {
        throw std::logic_error(fmt::format("Unexpected count of sizes for {}", device.str()));
    }
    return max_batch_sizes.front();
}

std::pair<int64_t, int64_t> CudaCaller::calculate_memory_requirements(
        const config::BasecallModelConfig &model_config) {
    // Determine size of working memory for CRFModel divided by (batch_size * chunk_size)
    // These values have been determined by running dorado with different models and
    // reporting the actual allocation size per chunk-timestep.
    int64_t crfmodel_bytes_per_chunk_timestep;
    if (model_config.is_flstm_model()) {
        if (model_config.lstm_size > 1024) {
            spdlog::warn("Unexpected model insize {}. Estimating GPU memory requirements.",
                         model_config.lstm_size);
        }
        crfmodel_bytes_per_chunk_timestep = 4096;
    } else if (model_config.out_features.has_value()) {
        auto out_features = model_config.out_features.value();
        const std::map<int, int64_t> out_features_map{{128, 2312}, {256, 8712}, {4096, 34848}};
        auto it = out_features_map.upper_bound(out_features - 1);
        if (it == out_features_map.end()) {
            spdlog::error(
                    "Failed to set GPU memory requirements. Unexpected model out_features {}.",
                    out_features);
            return {0, 0};
        } else if (it->first != out_features) {
            spdlog::warn("Unexpected model out_features {}. Estimating GPU memory requirements.");
        }
        crfmodel_bytes_per_chunk_timestep = it->second;
    } else {
        const std::map<int, int64_t> insize_map{
                {96, 960}, {128, 1280}, {384, 2816}, {768, 9728}, {1024, 10240}};
        auto it = insize_map.upper_bound(model_config.lstm_size - 1);
        if (it == insize_map.end()) {
            spdlog::error("Failed to set GPU memory requirements. Unexpected model insize {}.",
                          model_config.lstm_size);
            return {0, 0};
        } else if (it->first != model_config.lstm_size) {
            spdlog::warn("Unexpected model insize {}. Estimating GPU memory requirements.");
        }
        crfmodel_bytes_per_chunk_timestep = it->second;
    }

    // Determine size of working memory for decoder divided by (batch_size * chunk_size)
    // Decoder needs roughly (beam_width * 4) + num_states + 10 extra bytes
    // where num_states = 4^(state_len+1)
    // See `dorado::basecall::decode::CUDADecoder::beam_search_part_1()` for more details.
    int64_t decode_bytes_per_chunk_timestep =
            10 + default_beam_width * 4 + (1ull << (model_config.state_len * 2 + 2));

    return {crfmodel_bytes_per_chunk_timestep, decode_bytes_per_chunk_timestep};
}

CudaCaller::BatchDimsAndMaxSizes CudaCaller::calculate_batch_sizes(
        c10::Device device,
        float memory_limit_fraction,
        const config::BasecallModelConfig &model_config,
        std::optional<PipelineType> pipeline_type,
        int requested_batch_size,
        bool variable_chunk_sizes) {
    c10::cuda::CUDAGuard device_guard(device);
    c10::cuda::CUDACachingAllocator::emptyCache();
    const int batch_granularity = get_batch_size_granularity(model_config);
    const int chunk_granularity = model_config.chunk_size_granularity();
    const int stride = model_config.stride;
    const int min_chunk_size =
            utils::pad_to(model_config.basecaller.overlap() + 1, chunk_granularity);
    // Adjust chunk size to be a multiple of `chunk_granularity`, and greater than `overlap`.
    auto calculate_T_out = [=](int x) -> int {
        return std::max(min_chunk_size, (x / chunk_granularity) * chunk_granularity) / stride;
    };

    // First set of batch dimensions.
    const auto requested_chunk_size = model_config.basecaller.chunk_size();
    std::set<int> T_outs({calculate_T_out(requested_chunk_size)});

    // For high throughput simplex basecalling we use additional, shorter chunk sizes to handle
    // short reads better. As either of the queues might fill so slowly that we hit the
    // batch timeout and run partially filled batches, we set a long batch timeout.
    // As reads sitting in the pipeline for a long time doesn't mix well with duplex pairing,
    // we don't use extra chunk sizes for duplex. Similarly, for the low latency use case
    // (adaptive sampling) we only want one (short) chunk size so that all those reads go into
    // the same queue and complete as fast as possible.

    if (pipeline_type == PipelineType::simplex) {
        const char *env_extra_chunk_sizes = std::getenv("DORADO_EXTRA_CHUNK_SIZES");
        if (env_extra_chunk_sizes != nullptr) {
            constexpr char SEPARATOR = ';';
            std::string env_string(env_extra_chunk_sizes);
            for (size_t start = 0, end = 0; end != std::string::npos; start = end + 1) {
                T_outs.insert(calculate_T_out(std::atoi(env_string.c_str() + start)));
                end = env_string.find(SEPARATOR, start);
            }
        } else if (!variable_chunk_sizes) {
            // Use other chunk sizes as a fraction of the requested one
            // TODO: determine the best set of chunk sizes
            for (float fraction : {0.5f}) {
                T_outs.insert(calculate_T_out(int(requested_chunk_size * fraction)));
            }
        }
    }

    BatchDimsAndMaxSizes result;
    result.batch_dims.reserve(T_outs.size());
    for (auto iter = T_outs.rbegin(); iter != T_outs.rend(); ++iter) {
        result.batch_dims.push_back({
                .N = batch_granularity,
                .T_in = *iter * stride,
                .T_out = *iter,
        });
    }

    const int64_t gpu_mem_limit = get_gpu_mem_limit(device, memory_limit_fraction);
    spdlog::debug("{} memory limit {:.2f}GB", device.str(), gpu_mem_limit / GB);

    auto [crfmodel_bytes_per_ct, decode_bytes_per_ct] = calculate_memory_requirements(model_config);
    if (crfmodel_bytes_per_ct == 0) {
        return result;
    }

    // Batch size will be rounded up to a multiple of batch_size_granularity, regardless of
    // user choice. This makes sure batch size is compatible with GPU kernels.
    requested_batch_size = utils::pad_to(requested_batch_size, batch_granularity);
    result.max_batch_sizes.reserve(result.batch_dims.size());
    for (auto &batch_dim : result.batch_dims) {
        auto bytes_per_chunk = (crfmodel_bytes_per_ct + decode_bytes_per_ct) * batch_dim.T_out;
        int max_batch_size = int(gpu_mem_limit / bytes_per_chunk);
        max_batch_size -= max_batch_size % batch_granularity;
        if (max_batch_size < batch_granularity) {
            spdlog::warn(
                    "{} maximum safe estimated batch size at chunk size {} is only {}. Required "
                    "minimum is {}, GPU may run out of memory.",
                    device.str(), batch_dim.T_in, max_batch_size, batch_granularity);
            max_batch_size = batch_granularity;
        } else {
            spdlog::debug("{} maximum safe estimated batch size at chunk size {} is {}",
                          device.str(), batch_dim.T_in, max_batch_size);
        }

        result.max_batch_sizes.push_back(max_batch_size);

        if (requested_batch_size != 0) {
            if (requested_batch_size > max_batch_size) {
                spdlog::warn(
                        "{}: Requested batch size {} exceeds maximum safe estimated batch size {}.",
                        device.str(), requested_batch_size, max_batch_size);
            }
            batch_dim.N = requested_batch_size;
        }
    }

    return result;
}

void CudaCaller::determine_batch_dims(const BasecallerCreationParams &params,
                                      bool variable_chunk_sizes) {
    const int requested_batch_size = m_config.basecaller.batch_size();
    auto [batch_dims, max_batch_sizes] =
            calculate_batch_sizes(m_options.device(), params.memory_limit_fraction, m_config,
                                  m_pipeline_type, requested_batch_size, variable_chunk_sizes);
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
    const int batch_granularity = get_batch_size_granularity(m_config);
    const int stride = m_config.stride;

    // `288 * stride` (much shorter than the default chunk size of 10k), adjusted for
    // granularity, is a somewhat arbitrary trade-off between getting accurate measurements
    // and avoiding excessive startup time
    const int chunk_size = utils::pad_to(288 * stride, chunk_granularity);
    spdlog::debug("Auto batchsize {}: testing up to {} in steps of {}", m_device, max_batch_size,
                  batch_granularity);

    // Times and corresponding batch sizes.
    std::vector<std::pair<float, int>> times_and_batch_sizes;
    times_and_batch_sizes.reserve(max_batch_size / batch_granularity);

    const std::string model_name = m_config.model_path.filename().string();

    // See if we can find cached values for the chunk timings for this run condition
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    const auto chunk_benchmarks =
            ChunkBenchmarks::instance().get_chunk_timings(prop->name, model_name);
    if (!chunk_benchmarks) {
        spdlog::info(
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
                m_module->forward(input, aux.get());
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
                spdlog::trace("Auto batchsize {}: iteration:{}, ms/chunk {:8f} ms", m_device, i,
                              time_this_iteration);
            }
            // Clear the cache each time. Without this, intermittent cuda memory allocation errors
            // are seen on windows laptop NVIDIA RTX A5500 Laptop GPU. See JIRA issue DOR-466
            c10::cuda::CUDACachingAllocator::emptyCache();

            spdlog::debug("Auto batchsize {}: {}, time per chunk {:8f} ms", m_device, batch_size,
                          time);
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

        spdlog::debug(
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
    spdlog::debug("Largest batch size for {}: {}, time per chunk {:8f} ms", m_device,
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
        spdlog::debug("Final batch size for {}[{}]: {}", m_device, i, final_size);
    }
}

void CudaCaller::start_threads() {
    m_cuda_thread = std::thread([this] { cuda_thread_fn(); });
}

void CudaCaller::cuda_thread_fn() {
    utils::set_thread_name("cuda_caller");
    at::InferenceMode guard;
    const std::string loop_scope_str =
            "cuda_thread_fn_device_" + std::to_string(m_options.device().index());
    const std::string input_q_cv_scope_str =
            "input_queue_cv_device_" + std::to_string(m_options.device().index());

    c10::cuda::CUDAStreamGuard stream_guard(m_stream);
    auto &task_queue = get_task_queue();
    while (true) {
        nvtx3::scoped_range loop{loop_scope_str};
        std::unique_lock<std::mutex> input_lock(task_queue.m_input_lock);
        nvtxRangePushA(input_q_cv_scope_str.c_str());
        task_queue.m_input_cv.wait(input_lock, [&] {
            return (!task_queue.m_input_queue.empty() &&
                    task_queue.m_input_queue.front()->caller == this) ||
                   (task_queue.m_input_queue.empty() && m_terminate.load());
        });
        nvtxRangePop();

        if (task_queue.m_input_queue.empty() && m_terminate.load()) {
            return;
        }

        auto task = task_queue.m_input_queue.front();

        // pop and notify if this is a low latency queue so any other
        // low latency callers that are ready can get going immediately
        if (m_low_latency) {
            task_queue.m_input_queue.pop();
            input_lock.unlock();
            task_queue.m_input_cv.notify_all();
        }

        std::unique_lock<std::mutex> task_lock(task->mut);
        auto device_stats =
                c10::cuda::CUDACachingAllocator::getDeviceStats(m_options.device().index());

        const auto aggregate_idx =
                static_cast<uint64_t>(c10::CachingAllocator::StatType::AGGREGATE);
        spdlog::trace(
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
            at::Tensor signal_embeddings;
            auto scores = m_modbase_modules
                                  ? m_module->forward(task->input, task->aux, &signal_embeddings)
                                  : m_module->forward(task->input, task->aux);
            task->out = m_decoder->beam_search_part_1(
                    {scores, task->num_chunks, m_decoder_options, task->aux});
            if (m_modbase_modules) {
                run_modbase_models(*m_modbase_modules, task->out.data, signal_embeddings);
            }
            m_stream.synchronize();
            m_model_decode_ms += timer.GetElapsedMS();
        };

        try {
            run_basecalling();
        } catch (c10::Error &e) {
            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.msg());
            c10::cuda::CUDACachingAllocator::emptyCache();
            run_basecalling();
        }
        ++m_num_batches_called;
        task->done = true;
        task_lock.unlock();
        task->cv.notify_one();

        if (!m_low_latency) {
            // not low latency, so pop task and notify callers that we're ready to process a new one.
            // this prevents other callers that use the same GPU from attempting to call a new task
            // before this one has completed, without requiring an unsignalled mutex on the GPU
            // which was shown to have issues waking up threads in an even-handed manner
            task_queue.m_input_queue.pop();
            input_lock.unlock();
            task_queue.m_input_cv.notify_all();
        }
    }
}

}  // namespace dorado::basecall
