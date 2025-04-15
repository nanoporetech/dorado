#include "CudaCaller.h"

#include "benchmarks/CudaChunkBenchmarks.h"
#include "crf_utils.h"
#include "torch_utils/cuda_utils.h"
#include "utils/math_utils.h"
#include "utils/memory_utils.h"
#include "utils/thread_naming.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <limits>
#include <map>
#include <set>

using namespace std::chrono_literals;

namespace dorado::basecall {

namespace {

constexpr float GB = 1.0e9f;

void emit_benchmark_file(const std::string &gpu_name,
                         int compute_major,
                         const std::string &model,
                         const std::vector<std::pair<float, int>> &times_and_batch_sizes,
                         const std::vector<std::pair<float, int>> &all_times_and_batch_sizes) {
    // Prevent multiple devices outputting at once.
    static std::mutex batch_output_mutex;
    std::lock_guard<std::mutex> batch_output_lock(batch_output_mutex);

    std::string gpu_cuda_variant_name = gpu_name;
    // Hopper has specific optimizations that are only available if we are building with cuda12
    if (compute_major == 9 || compute_major == 10) {
        gpu_cuda_variant_name.append("_cuda");
        gpu_cuda_variant_name.append(std::to_string(CUDA_VERSION / 1000));
    }

    std::string cpp_filename = std::string("chunk_benchmarks__")
                                       .append(gpu_cuda_variant_name)
                                       .append("__")
                                       .append(model)
                                       .append(".txt");
    std::ofstream cpp_bench_file(cpp_filename);
    assert(cpp_bench_file);
    // Report out the batch sizes as a C++ map entry, for inclusion in dorado code
    cpp_bench_file << "    chunk_benchmarks[{\"" << gpu_name << "\", \"" << model << "\"}] = {\n";
    for (const auto &[batchsize, time] : times_and_batch_sizes) {
        cpp_bench_file << "        { " << time << ", " << batchsize << "f },\n";
    }
    cpp_bench_file << "    };\n";

    // Report out the batch sizes as a CSV file, for visualisation
    // For CSV output we output all timings, including ones which were worse than smaller batch sizes.
    std::string csv_filename = std::string("chunk_benchmarks__")
                                       .append(gpu_cuda_variant_name)
                                       .append("__")
                                       .append(model)
                                       .append(".csv");
    std::ofstream csv_bench_file(csv_filename);
    assert(csv_bench_file);
    csv_bench_file << "batch_size,time_per_chunk\n";
    for (const auto &[batchsize, time] : all_times_and_batch_sizes) {
        csv_bench_file << time << "," << batchsize << "\n";
    }
}

c10::cuda::CUDAStream get_stream_for_device(c10::Device device) {
    c10::cuda::CUDAGuard device_guard(device);
    return c10::cuda::getStreamFromPool(false, device.index());
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

struct CudaCaller::NNTask {
    NNTask(at::Tensor input_, int num_chunks_)
            : input(std::move(input_)), num_chunks(num_chunks_) {}
    at::Tensor input;
    int num_chunks;
    decode::DecodeData out;
    std::mutex mut;
    std::condition_variable cv;
    bool done{false};
};

CudaCaller::CudaCaller(const BasecallerCreationParams &params)
        : m_config(params.model_config),
          m_device(params.device),
          m_decoder(decode::create_decoder(params.device, m_config)),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(params.device)),
          m_low_latency(params.pipeline_type == PipelineType::simplex_low_latency),
          m_pipeline_type(params.pipeline_type),
          m_stream(get_stream_for_device(m_options.device())) {
    assert(m_options.device().is_cuda());
    assert(params.model_config.has_normalised_basecaller_params());

    m_decoder_options.q_shift = params.model_config.qbias;
    m_decoder_options.q_scale = params.model_config.qscale;
    m_num_input_features = params.model_config.num_features;

    at::InferenceMode guard;
    m_module = load_crf_model(params.model_config, m_options);

    determine_batch_dims(params);

    c10::cuda::CUDAGuard device_guard(m_options.device());
    c10::cuda::CUDACachingAllocator::emptyCache();

    auto [crfmodel_bytes_per_ct, decode_bytes_per_ct] = calculate_memory_requirements();

    // Warmup
    c10::cuda::CUDAStreamGuard stream_guard(m_stream);
    for (const auto &batch_dim : m_batch_dims) {
        spdlog::info("{} using chunk size {}, batch size {}", m_device, batch_dim.T_in,
                     batch_dim.N);
        spdlog::debug("{} Model memory {:.2f}GB", m_device,
                      (crfmodel_bytes_per_ct * batch_dim.T_out * batch_dim.N) / GB);
        spdlog::debug("{} Decode memory {:.2f}GB", m_device,
                      (decode_bytes_per_ct * batch_dim.T_out * batch_dim.N) / GB);
        auto input = torch::empty({batch_dim.N, m_num_input_features, batch_dim.T_in}, m_options);
        auto scores = m_module->forward(input);
        m_decoder->beam_search_part_1({scores, batch_dim.N, m_decoder_options});
    }
    m_stream.synchronize();

    start_threads();
}

CudaCaller::~CudaCaller() { terminate(); }

std::pair<int, int> CudaCaller::batch_timeouts_ms() const {
    // For low-latency pipelines we set both timeouts to the same value. This means that we
    // will always timeout based on the time from the first chunk being added to the batch.
    return m_low_latency
                   ? std::make_pair(DEFAULT_LOW_LATENCY_TIMEOUT_MS, DEFAULT_LOW_LATENCY_TIMEOUT_MS)
                   : std::make_pair(DEFAULT_FIRST_CHUNK_TIMEOUT_MS, DEFAULT_LAST_CHUNK_TIMEOUT_MS);
}

std::vector<decode::DecodedChunk> CudaCaller::call_chunks(at::Tensor &input,
                                                          at::Tensor &output,
                                                          int num_chunks) {
    NVTX3_FUNC_RANGE();
    if (num_chunks == 0) {
        return std::vector<decode::DecodedChunk>();
    }

    auto task = std::make_shared<NNTask>(input.to(m_options.device()), num_chunks);
    {
        std::lock_guard<std::mutex> lock(m_input_lock);
        m_input_queue.push_front(task);
    }
    m_input_cv.notify_one();

    std::unique_lock lock(task->mut);
    while (!task->done) {
        task->cv.wait(lock);
    }

    output.copy_(task->out.data);
    return m_decoder->beam_search_part_2({output, num_chunks, m_decoder_options});
}

void CudaCaller::terminate() {
    m_terminate.store(true);
    m_input_cv.notify_one();
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

std::pair<at::Tensor, at::Tensor> CudaCaller::create_input_output_tensor(
        size_t batch_dims_idx) const {
    auto opts = at::TensorOptions().device(torch::kCPU).pinned_memory(true);
    int64_t N = m_batch_dims[batch_dims_idx].N;
    int64_t T_in = m_batch_dims[batch_dims_idx].T_in;
    int64_t T_out = m_batch_dims[batch_dims_idx].T_out;
    int64_t C_in = m_num_input_features;
#if DORADO_TX2
    // The libtorch version on TX2 doesn't support `Tensor::view()` with a dtype of a different
    // size, so we use separate tensors here.
    auto input = torch::empty({N, C_in, T_in}, opts.dtype(m_options.dtype()));
    auto output = torch::empty({3, N, T_out}, opts.dtype(torch::kInt8));
#else
    auto scalar_type = c10::typeMetaToScalarType(m_options.dtype());
    // A runner's input and output buffers are never in use simultaneously, thus they can be mapped
    // to the same backing tensor.
    int64_t input_bytes = N * C_in * T_in * m_options.dtype().itemsize();
    int64_t output_bytes = 3 * N * T_out;
    auto storage = torch::empty({std::max(input_bytes, output_bytes)}, opts.dtype(torch::kInt8));
    auto input = storage.slice(0, 0, input_bytes).view(scalar_type).view({N, C_in, T_in});
    auto output = storage.slice(0, 0, output_bytes).view({3, N, T_out});
#endif
    return {input, output};
}

stats::NamedStats CudaCaller::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = static_cast<double>(m_num_batches_called);
    stats["model_decode_ms"] = static_cast<double>(m_model_decode_ms);
    return stats;
}

std::pair<int64_t, int64_t> CudaCaller::calculate_memory_requirements() const {
    // Determine size of working memory for CRFModel divided by (batch_size * chunk_size)
    // These values have been determined by running dorado with different models and
    // reporting the actual allocation size per chunk-timestep.
    int64_t crfmodel_bytes_per_chunk_timestep;
    if (m_config.out_features.has_value()) {
        auto out_features = m_config.out_features.value();
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
        auto it = insize_map.upper_bound(m_config.lstm_size - 1);
        if (it == insize_map.end()) {
            spdlog::error("Failed to set GPU memory requirements. Unexpected model insize {}.",
                          m_config.lstm_size);
            return {0, 0};
        } else if (it->first != m_config.lstm_size) {
            spdlog::warn("Unexpected model insize {}. Estimating GPU memory requirements.");
        }
        crfmodel_bytes_per_chunk_timestep = it->second;
    }

    // Determine size of working memory for decoder divided by (batch_size * chunk_size)
    // Decoder needs roughly (beam_width * 4) + num_states + 10 extra bytes
    // where num_states = 4^(state_len+1)
    // See `dorado::basecall::decode::CUDADecoder::beam_search_part_1()` for more details.
    int64_t decode_bytes_per_chunk_timestep =
            10 + m_decoder_options.beam_width * 4 + (1ull << (m_config.state_len * 2 + 2));

    return {crfmodel_bytes_per_chunk_timestep, decode_bytes_per_chunk_timestep};
}

void CudaCaller::determine_batch_dims(const BasecallerCreationParams &params) {
    auto requested_chunk_size = m_config.basecaller.chunk_size();
    auto requested_batch_size = m_config.basecaller.batch_size();

    c10::cuda::CUDAGuard device_guard(m_options.device());
    c10::cuda::CUDACachingAllocator::emptyCache();
    int64_t available = utils::available_memory(m_options.device());
    spdlog::debug("{} memory available: {:.2f}GB", m_device, available / GB);
    const int batch_granularity = get_batch_size_granularity(m_config);
    const int chunk_granularity = m_config.chunk_size_granularity();
    const int stride = m_config.stride;
    auto min_chunk_size = utils::pad_to(m_config.basecaller.overlap() + 1, chunk_granularity);
    // Adjust chunk size to be a multiple of `chunk_granularity`, and greater than `overlap`.
    auto calculate_T_out = [=](int x) -> int {
        return std::max(min_chunk_size, (x / chunk_granularity) * chunk_granularity) / stride;
    };

    // First set of batch dimensions.
    std::set<int> T_outs({calculate_T_out(requested_chunk_size)});
#if DORADO_TX2
    requested_batch_size = (requested_batch_size == 0) ? 256 : requested_batch_size;
    requested_batch_size = std::min(256, utils::pad_to(requested_batch_size, batch_granularity));
    int T_out = *(T_outs.begin());
    m_batch_dims.push_back({requested_batch_size, T_out * stride, T_out});
    return;
#endif

    // For high throughput simplex basecalling we use additional, shorter chunk sizes to handle
    // short reads better. As either of the queues might fill so slowly that we hit the
    // batch timeout and run partially filled batches, we set a long batch timeout.
    // As reads sitting in the pipeline for a long time doesn't mix well with duplex pairing,
    // we don't use extra chunk sizes for duplex. Similarly, for the low latency use case
    // (adaptive sampling) we only want one (short) chunk size so that all those reads go into
    // the same queue and complete as fast as possible.
    if (m_pipeline_type == PipelineType::simplex) {
        const char *env_extra_chunk_sizes = std::getenv("DORADO_EXTRA_CHUNK_SIZES");
        if (env_extra_chunk_sizes != nullptr) {
            constexpr char SEPARATOR = ';';
            std::string env_string(env_extra_chunk_sizes);
            for (size_t start = 0, end = 0; end != std::string::npos; start = end + 1) {
                T_outs.insert(calculate_T_out(std::atoi(env_string.c_str() + start)));
                end = env_string.find(SEPARATOR, start);
            }
        } else {
            // Use other chunk sizes as a fraction of the requested one
            // TODO: determine the best set of chunk sizes
            for (float fraction : {0.5f}) {
                T_outs.insert(calculate_T_out(int(requested_chunk_size * fraction)));
            }
        }
    }

    for (auto iter = T_outs.rbegin(); iter != T_outs.rend(); ++iter) {
        m_batch_dims.push_back({batch_granularity, *iter * stride, *iter});
    }

    // If running on a Jetson device with unified memory for CPU and GPU we can't use all
    // the available memory for GPU tasks. This way we leave at least half for the CPU,
    // though it's not clear what the ideal split would be.
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    bool is_unified_memory_device = (prop->major == 5 && prop->minor == 3) ||  // TX1
                                    (prop->major == 6 && prop->minor == 2) ||  // TX2
                                    (prop->major == 7 && prop->minor == 2) ||  // Xavier
                                    (prop->major == 8 && prop->minor == 7);    // Orin
    float memory_limit_fraction =
            params.memory_limit_fraction * (is_unified_memory_device ? 0.5f : 1.f);
    if (is_unified_memory_device && prop->major == 8 && available > (32 * GB)) {
        // restrict Orin further as there's no benefit to the largest batch sizes
        // and definite down sides to using all the memory
        memory_limit_fraction *= 0.5f;
    }

    // Apply limit fraction, and allow 1GB for model weights, etc.
    int64_t gpu_mem_limit = int64_t(available * memory_limit_fraction - GB);
    if (gpu_mem_limit < 0) {
        spdlog::warn("Failed to determine safe batch size. Less than 1GB GPU memory available.");
        return;
    }
    spdlog::debug("{} memory limit {:.2f}GB", m_device, gpu_mem_limit / GB);

    auto [crfmodel_bytes_per_ct, decode_bytes_per_ct] = calculate_memory_requirements();
    if (crfmodel_bytes_per_ct == 0) {
        return;
    }

    // Batch size will be rounded up to a multiple of batch_size_granularity, regardless of
    // user choice. This makes sure batch size is compatible with GPU kernels.
    requested_batch_size = utils::pad_to(requested_batch_size, batch_granularity);
    std::vector<int> max_batch_sizes;
    for (auto &batch_dim : m_batch_dims) {
        auto bytes_per_chunk = (crfmodel_bytes_per_ct + decode_bytes_per_ct) * batch_dim.T_out;
        int max_batch_size = int(gpu_mem_limit / bytes_per_chunk);
        max_batch_size -= max_batch_size % batch_granularity;
        if (max_batch_size < batch_granularity) {
            spdlog::warn(
                    "{} maximum safe estimated batch size at chunk size {} is only {}. Required "
                    "minimum is {}, GPU may run out of memory.",
                    m_device, batch_dim.T_in, max_batch_size, batch_granularity);
            max_batch_size = batch_granularity;
        } else {
            spdlog::debug("{} maximum safe estimated batch size at chunk size {} is {}", m_device,
                          batch_dim.T_in, max_batch_size);
        }

        if (requested_batch_size == 0) {
            max_batch_sizes.push_back(max_batch_size);
        } else {
            if (requested_batch_size > max_batch_size) {
                spdlog::warn(
                        "{}: Requested batch size {} exceeds maximum safe estimated batch size {}.",
                        m_device, requested_batch_size, max_batch_size);
            }
            batch_dim.N = std::min(requested_batch_size, max_batch_size);
        }
    }

    if (requested_batch_size != 0) {
        return;
    }

    float best_time = std::numeric_limits<float>::max();

    assert(m_batch_dims.size() > 0);
    // We limit the maximum when doing benchmarking to avoid excessive startup time.
    // The limit for transformer models should be increased at a later time.
    int max_batch_size = *std::max_element(max_batch_sizes.begin(), max_batch_sizes.end());
    const int max_batch_size_limit = m_config.is_tx_model() ? 1024 : 10240;
    max_batch_size = std::min(max_batch_size, max_batch_size_limit);

    // When we are emitting benchmarks, prefer accuracy to speed of benchmark generation, so
    // run the benchmarks at full chunk size.
    int chunk_size = m_batch_dims.back().T_in;
    if (!params.emit_batchsize_benchmarks) {
        // `288 * stride` (much shorter than the default chunk size of 10k), adjusted for
        // granularity, is a somewhat arbitrary trade-off between getting accurate measurements
        // and avoiding excessive startup time
        chunk_size = utils::pad_to(288 * stride, chunk_granularity);
    }
    spdlog::debug("Auto batchsize {}: testing up to {} in steps of {}", m_device, max_batch_size,
                  batch_granularity);

    // Times and corresponding batch sizes.
    std::vector<std::pair<float, int>> times_and_batch_sizes;
    std::vector<std::pair<float, int>> all_times_and_batch_sizes;
    times_and_batch_sizes.reserve(max_batch_size / batch_granularity);

    const std::string model_name = m_config.model_path.filename().string();

    // See if we can find cached values for the chunk timings for this run condition
    const auto chunk_benchmarks =
            CudaChunkBenchmarks::instance().get_chunk_timings(prop->name, model_name);
    if (!chunk_benchmarks || params.run_batchsize_benchmarks) {
        spdlog::info(
                "Calculating optimized batch size for GPU \"{}\" and model {}. Full benchmarking "
                "will run for this device, which may take some time.",
                prop->name, model_name);
    }

    for (int batch_size = batch_granularity; batch_size <= max_batch_size;
         batch_size += batch_granularity) {
        float time = std::numeric_limits<float>::max();

        // Use the available cached chunk size if we haven't been explicitly told not to.
        if (!params.run_batchsize_benchmarks && chunk_benchmarks) {
            // Note that if a cache of batch size timings is available, we don't mix cached and live
            //  benchmarks, to avoid discontinuities in the data.
            if (chunk_benchmarks->find(batch_size) != chunk_benchmarks->end()) {
                time = chunk_benchmarks->at(batch_size);
            }
        } else {
            auto input = torch::empty({batch_size, m_config.num_features, chunk_size}, m_options);

            for (int i = 0; i < 2; ++i) {  // run twice to eliminate outliers
                using utils::handle_cuda_result;
                cudaEvent_t start, stop;
                handle_cuda_result(cudaEventCreate(&start));
                handle_cuda_result(cudaEventCreate(&stop));
                handle_cuda_result(cudaEventRecord(start));
                m_module->forward(input);
                handle_cuda_result(cudaEventRecord(stop));
                handle_cuda_result(cudaEventSynchronize(stop));
                float ms = 0;
                handle_cuda_result(cudaEventElapsedTime(&ms, start, stop));
                auto time_this_iteration = ms / batch_size;
                time = std::min(time, time_this_iteration);
                handle_cuda_result(cudaEventDestroy(start));
                handle_cuda_result(cudaEventDestroy(stop));
                spdlog::trace("Auto batchsize {}: iteration:{}, ms/chunk {:8f} ms", m_device, i,
                              time_this_iteration);
            }
            // Clear the cache each time. Without this, intermittent cuda memory allocation errors
            // are seen on windows laptop NVIDIA RTX A5500 Laptop GPU. See JIRA issue DOR-466
            c10::cuda::CUDACachingAllocator::emptyCache();

            spdlog::debug("Auto batchsize {}: {}, time per chunk {:8f} ms", m_device, batch_size,
                          time);
        }

        all_times_and_batch_sizes.emplace_back(time, batch_size);
        if (time < best_time) {
            best_time = time;
            times_and_batch_sizes.emplace_back(time, batch_size);
        }
    }

    if (params.emit_batchsize_benchmarks) {
        emit_benchmark_file(prop->name, prop->major, model_name, times_and_batch_sizes,
                            all_times_and_batch_sizes);
    }

    if (!chunk_benchmarks) {
        // If we have just generated benchmarks that didn't previously exist, add them to the in-memory cache. This
        // will be of benefit to basecall servers which won't have to keep re-generating the benchmarks each time a
        // runner is created.
        CudaChunkBenchmarks::instance().add_chunk_timings(prop->name, model_name,
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
    const std::string gpu_lock_scope_str = "gpu_lock_" + std::to_string(m_options.device().index());

    c10::cuda::CUDAStreamGuard stream_guard(m_stream);
    while (true) {
        nvtx3::scoped_range loop{loop_scope_str};
        std::unique_lock<std::mutex> input_lock(m_input_lock);
        nvtxRangePushA(input_q_cv_scope_str.c_str());
        while (m_input_queue.empty() && !m_terminate.load()) {
            m_input_cv.wait_for(input_lock, 100ms);
        }
        nvtxRangePop();

        if (m_input_queue.empty() && m_terminate.load()) {
            return;
        }

        auto task = m_input_queue.back();
        m_input_queue.pop_back();
        input_lock.unlock();

        nvtxRangePushA(gpu_lock_scope_str.c_str());
        auto gpu_lock = dorado::utils::acquire_gpu_lock(m_options.device().index(), !m_low_latency);
        nvtxRangePop();

        std::unique_lock<std::mutex> task_lock(task->mut);

        auto device_stats =
                c10::cuda::CUDACachingAllocator::getDeviceStats(m_options.device().index());

        auto print_stat = [](const auto &st) {
            return "aggregate current " + std::to_string(st[0].current);
        };
        spdlog::trace(
                "allocation {}, segment {}, active {}, inactive_split {}, alloc_bytes {}, "
                "reserved_bytes {}, active_bytes {}, inactive_split_bytes {}, requested_bytes "
                "{}, num_alloc_retries {}, num_ooms {}, max_split_size {}",
                print_stat(device_stats.allocation), print_stat(device_stats.segment),
                print_stat(device_stats.active), print_stat(device_stats.inactive_split),
                print_stat(device_stats.allocated_bytes), print_stat(device_stats.reserved_bytes),
                print_stat(device_stats.active_bytes),
                print_stat(device_stats.inactive_split_bytes),
#if TORCH_VERSION_MAJOR >= 2
                print_stat(device_stats.requested_bytes),
#else
                "unknown",
#endif  // TORCH_VERSION_MAJOR > 1
                device_stats.num_alloc_retries, device_stats.num_alloc_retries,
                device_stats.num_ooms, device_stats.max_split_size);

        auto run_basecalling = [&]() {
            stats::Timer timer;
            auto scores = m_module->forward(task->input);
            task->out =
                    m_decoder->beam_search_part_1({scores, task->num_chunks, m_decoder_options});
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
    }
}

}  // namespace dorado::basecall
