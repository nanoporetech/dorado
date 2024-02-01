#include "CudaCaller.h"

#include "crf_utils.h"
#include "utils/cuda_utils.h"
#include "utils/math_utils.h"
#include "utils/memory_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>

using namespace std::chrono_literals;

namespace dorado::basecall {

static constexpr float GB = 1.0e9f;

struct CudaCaller::NNTask {
    NNTask(at::Tensor input_, int num_chunks_) : input(input_), num_chunks(num_chunks_) {}
    at::Tensor input;
    int num_chunks;
    decode::DecodeData out;
    std::mutex mut;
    std::condition_variable cv;
    bool done{false};
};

CudaCaller::CudaCaller(const CRFModelConfig &model_config,
                       int chunk_size,
                       int batch_size,
                       const std::string &device,
                       float memory_limit_fraction,
                       bool exclusive_gpu_access,
                       bool low_latency)
        : m_config(model_config),
          m_device(device),
          m_decoder(decode::create_decoder(device, m_config)),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(device)),
          m_exclusive_gpu_access(exclusive_gpu_access),
          m_low_latency(low_latency) {
    assert(m_options.device().is_cuda());

    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;
    m_num_input_features = model_config.num_features;

    at::InferenceMode guard;
    m_module = load_crf_model(model_config, m_options);

    determine_batch_sizes(memory_limit_fraction, batch_size, chunk_size);

    c10::cuda::CUDAGuard device_guard(m_options.device());
    c10::cuda::CUDACachingAllocator::emptyCache();

    auto [crfmodel_bytes_per_ct, decode_bytes_per_ct] = calculate_memory_requirements();

    // Warmup
    for (size_t i = 0; i < m_batch_sizes.size(); ++i) {
        auto input = torch::empty({m_batch_sizes[i], m_num_input_features, m_in_chunk_sizes[i]},
                                  m_options);
        auto scores = m_module->forward(input);
        m_decoder->beam_search_part_1({scores, m_batch_sizes[i], m_decoder_options});
        spdlog::info("{} using chunk size {}: {}, batch size {}", m_device, i, m_in_chunk_sizes[i],
                     m_batch_sizes[i]);
        spdlog::debug("{} Model memory {:.2f}GB", m_device,
                      (crfmodel_bytes_per_ct * m_out_chunk_sizes[i] * m_batch_sizes[i]) / GB);
        spdlog::debug("{} Decode memory {:.2f}GB", m_device,
                      (decode_bytes_per_ct * m_out_chunk_sizes[i] * m_batch_sizes[i]) / GB);
    }
    torch::cuda::synchronize(m_options.device().index());

    start_threads();
}

CudaCaller::~CudaCaller() {
    m_terminate.store(true);
    m_input_cv.notify_one();
    if (m_cuda_thread && m_cuda_thread->joinable()) {
        m_cuda_thread->join();
    }
}

std::vector<decode::DecodedChunk> CudaCaller::call_chunks(at::Tensor &input,
                                                          at::Tensor &output,
                                                          int num_chunks,
                                                          c10::cuda::CUDAStream stream) {
    NVTX3_FUNC_RANGE();
    c10::cuda::CUDAStreamGuard stream_guard(stream);

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
    if (m_cuda_thread && m_cuda_thread->joinable()) {
        m_cuda_thread->join();
    }
    m_cuda_thread.reset();
}

void CudaCaller::restart() {
    // This can be called more than one, via multiple runners.
    if (m_terminate.load()) {
        m_terminate.store(false);
        start_threads();
    }
}

std::pair<at::Tensor, at::Tensor> CudaCaller::create_input_output_tensor(int chunk_size_idx) const {
    auto opts = at::TensorOptions().device(torch::kCPU).pinned_memory(true);
    int64_t N = m_batch_sizes[chunk_size_idx];
    int64_t T_in = m_in_chunk_sizes[chunk_size_idx];
    int64_t T_out = m_out_chunk_sizes[chunk_size_idx];
    int64_t C_in = m_num_input_features;
#ifdef DORADO_TX2
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
    stats["batches_called"] = double(m_num_batches_called);
    stats["model_ms"] = double(m_model_ms);
    stats["decode_ms"] = double(m_decode_ms);
    return stats;
}

std::pair<int64_t, int64_t> CudaCaller::calculate_memory_requirements() const {
    // Determine size of working memory for CRFModel divided by (batch_size * chunk_size)
    // These values have been determined by running dorado with different models and
    // reporting the actual allocation size per chunk-timestep.
    int64_t crfmodel_bytes_per_chunk_timestep;
    if (m_config.out_features.has_value()) {
        auto out_features = m_config.out_features.value();
        std::unordered_map<int, int64_t> out_features_map{{128, 2312}, {256, 8712}};
        crfmodel_bytes_per_chunk_timestep = out_features_map[out_features];
        if (crfmodel_bytes_per_chunk_timestep == 0) {
            spdlog::warn("Failed to set GPU memory requirements. Unexpected model out_features {}.",
                         out_features);
            return {0, 0};
        }
    } else {
        std::unordered_map<int, int64_t> insize_map{
                {96, 960}, {128, 1280}, {384, 2816}, {768, 9728}, {1024, 10240}};
        crfmodel_bytes_per_chunk_timestep = insize_map[m_config.lstm_size];
        if (crfmodel_bytes_per_chunk_timestep == 0) {
            spdlog::warn("Failed to set GPU memory requirements. Unexpected model insize {}.",
                         m_config.lstm_size);
            return {0, 0};
        }
    }

    // Determine size of working memory for decoder divided by (batch_size * chunk_size)
    // Decoder needs roughly (beam_width * 4) + num_states + 10 extra bytes
    // where num_states = 4^(state_len+1)
    // See `dorado::basecall::decode::CUDADecoder::beam_search_part_1()` for more details.
    int64_t decode_bytes_per_chunk_timestep =
            10 + m_decoder_options.beam_width * 4 + (1ull << (m_config.state_len * 2 + 2));

    return {crfmodel_bytes_per_chunk_timestep, decode_bytes_per_chunk_timestep};
}

void CudaCaller::determine_batch_sizes(float memory_limit_fraction,
                                       int requested_batch_size,
                                       int requested_chunk_size) {
    c10::cuda::CUDAGuard device_guard(m_options.device());
    int64_t available = utils::available_memory(m_options.device());
    spdlog::debug("{} memory available: {:.2f}GB", m_device, available / GB);

    // Adjust chunk size to be a multiple of the stride
    m_out_chunk_sizes.push_back(requested_chunk_size / m_config.stride);
    m_in_chunk_sizes.push_back(m_out_chunk_sizes[0] * m_config.stride);
#ifdef DORADO_TX2
    m_batch_sizes.push_back(256);
    return;
#endif

    if (!m_low_latency) {
        // Use a second chunk size which is half the requested one
        m_out_chunk_sizes.push_back(m_out_chunk_sizes[0] / 2);
        m_in_chunk_sizes.push_back(m_out_chunk_sizes[1] * m_config.stride);
    }

    const int granularity = get_batch_size_granularity();
    m_batch_sizes = std::vector<int>(m_in_chunk_sizes.size(), granularity);

    // If running on a Jetson device with unified memory for CPU and GPU we can't use all
    // the available memory for GPU tasks. This way we leave at least half for the CPU,
    // though it's not clear what the ideal split would be.
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    bool is_unified_memory_device = (prop->major == 5 && prop->minor == 3) ||  // TX1
                                    (prop->major == 6 && prop->minor == 2) ||  // TX2
                                    (prop->major == 7 && prop->minor == 2) ||  // Xavier
                                    (prop->major == 8 && prop->minor == 7);    // Orin
    memory_limit_fraction *= is_unified_memory_device ? 0.5f : 1.f;

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
    requested_batch_size = utils::pad_to(requested_batch_size, granularity);
    for (size_t i = 0; i < m_out_chunk_sizes.size(); ++i) {
        auto bytes_per_chunk = (crfmodel_bytes_per_ct + decode_bytes_per_ct) * m_out_chunk_sizes[i];
        int max_batch_size = int(gpu_mem_limit / bytes_per_chunk);
        max_batch_size -= max_batch_size % granularity;
        if (max_batch_size <= granularity) {
            spdlog::warn("{} maximum safe estimated batch size at chunk size {} is only {}.",
                         m_device, m_in_chunk_sizes[i], max_batch_size);
            continue;
        }
        spdlog::debug("{} maximum safe estimated batch size at chunk size {} is {}", m_device,
                      m_in_chunk_sizes[i], max_batch_size);
        if (requested_batch_size == 0) {
            m_batch_sizes[i] = max_batch_size;
        } else {
            if (requested_batch_size > max_batch_size) {
                spdlog::warn(
                        "{}: Requested batch size {} exceeds maximum safe estimated batch size {}.",
                        m_device, requested_batch_size, max_batch_size);
            }
            m_batch_sizes[i] = std::min(requested_batch_size, max_batch_size);
        }
    }

    if (requested_batch_size != 0) {
        return;
    }

    auto max_batch_sizes = m_batch_sizes;
    float best_time = std::numeric_limits<float>::max();
    const int chunk_size = std::min(m_in_chunk_sizes.back(), m_config.stride * 300);
    // We limit the maximum when doing benchmarking to avoid excessive startup time.
    const int max_batch_size_limit = 10240;
    int max_batch_size = *std::max_element(max_batch_sizes.begin(), max_batch_sizes.end());
    max_batch_size = std::min(max_batch_size, max_batch_size_limit);
    spdlog::debug("Auto batchsize {}: testing up to {} in steps of {}", m_device, max_batch_size,
                  granularity);
    for (int batch_size = granularity; batch_size <= max_batch_size; batch_size += granularity) {
        auto input = torch::empty({batch_size, m_config.num_features, chunk_size}, m_options);

        float time = std::numeric_limits<float>::max();
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
            time = std::min(time, ms / batch_size);
            handle_cuda_result(cudaEventDestroy(start));
            handle_cuda_result(cudaEventDestroy(stop));
        }

        spdlog::debug("Auto batchsize {}: {}, time per chunk {:8f} ms", m_device, batch_size, time);
        if (time < best_time) {
            best_time = time;
            for (size_t i = 0; i < m_batch_sizes.size(); ++i) {
                if (batch_size <= max_batch_sizes[i]) {
                    m_batch_sizes[i] = batch_size;
                }
            }
        }

        // Clear the cache each time. Without this, intermittent cuda memory allocation errors
        // are seen on windows laptop NVIDIA RTX A5500 Laptop GPU. See JIRA issue DOR-466
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
}

void CudaCaller::start_threads() {
    m_cuda_thread.reset(new std::thread(&CudaCaller::cuda_thread_fn, this));
}

void CudaCaller::cuda_thread_fn() {
    at::InferenceMode guard;
    c10::cuda::CUDAGuard device_guard(m_options.device());
    auto stream = c10::cuda::getCurrentCUDAStream(m_options.device().index());

    const std::string loop_scope_str =
            "cuda_thread_fn_device_" + std::to_string(m_options.device().index());
    const std::string input_q_cv_scope_str =
            "input_queue_cv_device_" + std::to_string(m_options.device().index());
    const std::string gpu_lock_scope_str = "gpu_lock_" + std::to_string(m_options.device().index());
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
        auto gpu_lock =
                dorado::utils::acquire_gpu_lock(m_options.device().index(), m_exclusive_gpu_access);
        nvtxRangePop();

        std::unique_lock<std::mutex> task_lock(task->mut);

        auto device_stats =
                c10::cuda::CUDACachingAllocator::getDeviceStats(m_options.device().index());

        auto print_stat = [](c10::cuda::CUDACachingAllocator::StatArray &st) {
            std::string s("");
            s += "aggregate current " + std::to_string(st[0].current);
            s += "\n";
            return s;
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
            const auto forward_ms = timer.GetElapsedMS();
            task->out =
                    m_decoder->beam_search_part_1({scores, task->num_chunks, m_decoder_options});
            stream.synchronize();
            const auto forward_plus_decode_ms = timer.GetElapsedMS();
            m_model_ms += forward_ms;
            m_decode_ms += forward_plus_decode_ms - forward_ms;
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
