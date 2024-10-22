#include "MetalCaller.h"

#include "ModelRunnerBase.h"
#include "crf_utils.h"
#include "decode/beam_search.h"
#include "nn/TxModel.h"
#include "torch_utils/metal_utils.h"
#include "utils/math_utils.h"
#include "utils/memory_utils.h"
#include "utils/thread_naming.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>

using namespace dorado::utils;
using namespace std::chrono_literals;
using torch::indexing::Slice;

namespace {
constexpr int MTL_CORE_BATCH_SIZE = 48;
}  // namespace

namespace dorado::basecall {

struct MetalCaller::NNTask {
    NNTask(at::Tensor *input_, int num_chunks_, std::vector<decode::DecodedChunk> *out_chunks_)
            : input(input_), out_chunks(out_chunks_), num_chunks(num_chunks_) {}

    // LSTM: NTC - Tx: NCT
    at::Tensor *input;
    std::mutex mut;
    std::condition_variable cv;
    std::vector<decode::DecodedChunk> *out_chunks;
    int num_chunks;
    int decode_chunks_started{0};
    int decode_chunks_finished{0};
    // Event ID to be signalled when decoding for this task is complete, set by metal_thread_fn.
    uint64_t decode_complete_event_id = static_cast<uint64_t>(0);
};

MetalCaller::~MetalCaller() { terminate(); }

void MetalCaller::call_chunks(at::Tensor &input,
                              int num_chunks,
                              std::vector<decode::DecodedChunk> &out_chunks) {
    if (num_chunks == 0) {
        return;
    }

    // Input can be NTC or NCT for LSTM and Tx models respectively
    auto task = std::make_shared<NNTask>(&input, num_chunks, &out_chunks);
    {
        std::lock_guard<std::mutex> lock(m_input_lock);
        m_input_queue.push_front(task);
    }
    m_input_cv.notify_one();

    std::unique_lock lock(task->mut);
    while (task->decode_chunks_finished != num_chunks) {
        task->cv.wait(lock);
    }
}

void MetalCaller::terminate() {
    m_terminate.store(true);
    m_input_cv.notify_one();
    m_decode_cv.notify_all();
    if (m_metal_thread.joinable()) {
        m_metal_thread.join();
    }
    for (auto &thr : m_decode_threads) {
        thr.join();
    }
    m_decode_threads.clear();
}

void MetalCaller::restart() {
    // This can be called more than once, via multiple runners.
    if (m_terminate.exchange(false)) {
        m_terminate_decode.store(false);
        start_threads();
    }
}

void MetalCaller::start_threads() {
    m_metal_thread = std::thread([this] { metal_thread_fn(); });

    int num_decode_threads = std::max(1, get_apple_cpu_perf_core_count() - 1);
    m_decode_threads.reserve(num_decode_threads);
    for (int i = 0; i < num_decode_threads; ++i) {
        m_decode_threads.emplace_back([this] { decode_thread_fn(); });
    }
}

void MetalCaller::metal_thread_fn() {
    utils::set_thread_name("metal_worker");
    at::InferenceMode inference_mode_guard;
    ScopedAutoReleasePool outer_pool;

    // Incrementing ID used to prevent the linear layer of run i+1 overwriting the scores of
    // run i before the CPU has finished decoding all run i's chunks.
    // Start at 1, since at event creation ID 0 is deemed to have been signalled.
    auto next_decode_complete_event_id = static_cast<uint64_t>(1);

    // For unknown reasons, concurrent access to the GPU from multiple instances of this thread --
    // i.e. with > 1 instance of MetalCaller -- results in errors, usually command buffer error code 1.
    // Holding this mutex while executing models seemingly prevents these errors.
    static std::mutex inter_caller_mutex;

    while (true) {
        ScopedAutoReleasePool inner_pool;

        // Pop the next task, or return if we're terminated.
        std::shared_ptr<NNTask> task;
        {
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            while (m_input_queue.empty() && !m_terminate.load()) {
                m_input_cv.wait_for(input_lock, 100ms);
            }

            if (m_input_queue.empty() && m_terminate.load()) {
                m_terminate_decode.store(true);
                return;
            }

            task = std::move(m_input_queue.back());
            m_input_queue.pop_back();
        }

        // Assign this task a unique decode completion event ID.
        // This ID will be signalled by the CPU once it has finished relevant decoding work,
        // allowing the GPU to proceed.
        task->decode_complete_event_id = next_decode_complete_event_id++;

        // Basecall the chunk and run the scan kernels on GPU
        {
            auto retry_delay = 100ms;
            auto sleep_before_retry = [&] {
                std::this_thread::sleep_for(retry_delay);
                retry_delay *= 2;
                // These are rare enough that sleeping for a few seconds shouldn't impact
                // speed, and should give the system plenty of time to recover.
                if (retry_delay > 5s) {
                    retry_delay = 5s;
                }
            };

            // We retry the entire set of kernels up to 5 times, to deal with seemingly
            // random intermittent errors with command buffer submissions.
            // On iOS we can't bail here, so we need to keep retrying.
            // TODO: find a more robust way of dealing with Metal kernel launch issues
            bool cb_success = false;
#if TARGET_OS_IPHONE
            for (int try_count = 0; !cb_success; ++try_count)
#else   // TARGET_OS_IPHONE
            for (int try_count = 0; try_count < 5; ++try_count)
#endif  // TARGET_OS_IPHONE
            {
                cb_success = call_task(*task, inter_caller_mutex, try_count);
                if (cb_success) {
                    break;
                } else {
                    sleep_before_retry();
                }
            }

            // If we repeatedly submitted CBs without success, we give up.
            if (!cb_success) {
                spdlog::critical("Exiting. Failed to successfully submit GPU command buffers.");
                std::exit(EXIT_FAILURE);
            }
        }

        // Pass task on to decode threads
        {
            std::lock_guard decode_lock(m_decode_lock);
            m_decode_queue.push_front(std::move(task));
        }
        m_decode_cv.notify_all();
    }
}

void MetalCaller::decode_thread_fn() {
    utils::set_thread_name("metal_decode");
    at::InferenceMode inference_mode_guard;
    while (true) {
        std::unique_lock<std::mutex> decode_lock(m_decode_lock);
        while (m_decode_queue.empty() && !m_terminate_decode.load()) {
            m_decode_cv.wait_for(decode_lock, 100ms);
        }

        if (m_decode_queue.empty() && m_terminate_decode.load()) {
            return;
        }
        auto task = m_decode_queue.back();
        int chunk_idx = task->decode_chunks_started++;
        // If all chunks have been picked up for decoding, remove task from queue
        if (chunk_idx == task->num_chunks - 1) {
            m_decode_queue.pop_back();
        }
        decode_lock.unlock();

        auto [sequence, qstring, moves] = decode(chunk_idx);
        (*task->out_chunks)[chunk_idx] =
                decode::DecodedChunk{std::move(sequence), std::move(qstring), std::move(moves)};

        // Wake the waiting thread which called `call_chunks()` if we're done decoding
        std::unique_lock<std::mutex> task_lock(task->mut);
        bool done = ++(task->decode_chunks_finished) == task->num_chunks;
        task_lock.unlock();
        if (done) {
            if (m_decode_complete_event) {
                // Now that all chunks are decoded, signal that the GPU can overwrite the scores
                // buffer with subsequent work.
                m_decode_complete_event->setSignaledValue(task->decode_complete_event_id);
            }
            task->cv.notify_one();
        }
    }
}

MetalLSTMCaller::MetalLSTMCaller(const CRFModelConfig &model_config, float memory_limit_fraction)
        : MetalCaller(model_config) {
    ScopedAutoReleasePool autorelease_pool;
    // Our metal builds assume shared memory, so it's safe to check host.
    if (auto total_mem = utils::total_host_memory_GB(); total_mem < 16) {
        spdlog::warn(
                "Less than 16GB of memory available: {}GB detected. "
                "This is below minimum spec and may cause issues",
                total_mem);
    }

    m_device = get_mtl_device();

    m_decoder_options = decode::DecoderOptions();
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;

    // TODO -- we don't honour the config n_base
    constexpr int n_base = 4;
    m_states = pow(n_base, model_config.state_len);

    auto state_dict = load_crf_model_weights(model_config);

    assert(model_config.has_normalised_basecaller_params());
    const auto chunk_size = model_config.basecaller.chunk_size();
    const auto batch_size = model_config.basecaller.batch_size();
    auto selected_batch_size = (batch_size == 0) ? benchmark_batch_sizes(model_config, state_dict,
                                                                         memory_limit_fraction)
                                                 : utils::pad_to(batch_size, MTL_CORE_BATCH_SIZE);

    set_chunk_batch_size(model_config, state_dict, chunk_size, selected_batch_size);
    start_threads();
}

void MetalLSTMCaller::set_chunk_batch_size(const CRFModelConfig &model_config,
                                           const std::vector<at::Tensor> &state_dict,
                                           int chunk_size,
                                           int batch_size) {
    // Chunk size already normalised to inner stride
    m_in_chunk_size = chunk_size;
    m_out_chunk_size = chunk_size / model_config.stride;

    m_batch_size = batch_size;

    // Allocations beyond 4GB can fail, and the linear layer output buffer
    // hits this limit with batch sizes larger than 384 with typical
    // chunk sizes.  We also want to limit memory usage in general.
    // At the same time, the LSTM layer performance benefits
    // from large batch sizes.
    // We therefore run the linear layer via 1 or more kernel runs, each
    // with an output buffer of limited size, aiming for <= kMaxBufferSize.
    // As things stand, we need an exactly even split of batch elements in
    // the linear layer output buffers (this could be relaxed).
    // We therefore want the smallest divisor of batch_size that results in
    // linear layer output buffers <= kMaxBufferSize, and a linear layer batch size
    // that is an integral multiple of 48.  Since the LSTM batch size is
    // already constrained to be an integral multiple of 48, this means the
    // batch splitting factor must be an exact divisor of the batch_size / 48.

    // If this target is smaller than the size required for 48 batch elements, then
    // that size is the best we can do.  The size here is attainable for fast and hac
    // models, but not sup.
    constexpr auto kMaxBufferSize = static_cast<int64_t>(1) << 29;
    const auto complete_linear_out_size =
            static_cast<int64_t>(m_out_chunk_size) * static_cast<int64_t>(m_batch_size) *
            static_cast<int64_t>(model_config.outsize) * sizeof(float);
    const int num_batch_pieces = m_batch_size / MTL_CORE_BATCH_SIZE;
    for (m_out_split = 1; m_out_split < num_batch_pieces; ++m_out_split) {
        if (num_batch_pieces % m_out_split == 0 &&
            complete_linear_out_size / m_out_split <= kMaxBufferSize) {
            break;
        }
    }
    auto piece_size = complete_linear_out_size / m_out_split;
    if (piece_size > kMaxBufferSize) {
        spdlog::debug("Did not hit linear layer target output size {} - got {}", kMaxBufferSize,
                      piece_size);
    }
    spdlog::debug("Linear layer split {}", m_out_split);
    // If we exited the loop above without breaking, then m_out_split = num_batch_pieces,
    // which satisfies the divisor criterion, and should mean small enough linear layer
    // output buffers, given other reasonable parameters.
    assert(num_batch_pieces % m_out_split == 0);
    assert(m_batch_size % m_out_split == 0);
    m_out_batch_size = m_batch_size / m_out_split;
    assert(m_out_batch_size % MTL_CORE_BATCH_SIZE == 0);

    m_model = nn::MetalCRFModel(model_config, m_in_chunk_size, m_batch_size, m_out_split,
                                m_device.get());
    m_model->load_state_dict(state_dict);
    m_model->eval();

    m_decode_complete_event = NS::TransferPtr(m_device->newSharedEvent());
    m_bwd_scan_cps = make_cps(m_device.get(), "backward_scan", {}, std::nullopt);
    m_fwd_scan_add_softmax_cps =
            make_cps(m_device.get(), "forward_scan_add_softmax", {}, std::nullopt);

    int T = m_out_chunk_size;
    int C = model_config.outsize;
    int Cs = m_states;

    m_scores_TNC.clear();
    m_posts_NTC.clear();
    m_bwd_NTC.clear();
    for (int i = 0; i < m_out_split; ++i) {
        m_scores_TNC.push_back(torch::empty({T, m_out_batch_size, C}, m_scores_dtype));
        // Unfortunately torch doesn't have Uint16, or we would use it.  We could offset,
        // or rely on undefined overflow behaviour, but for now we waste the sign bit.
        m_posts_NTC.push_back(torch::empty({m_out_batch_size, T + 1, Cs}, m_posts_dtype));
        m_bwd_NTC.push_back(torch::empty({m_out_batch_size, T + 1, Cs}));
    }
}

int MetalLSTMCaller::benchmark_batch_sizes(const CRFModelConfig &model_config,
                                           const std::vector<at::Tensor> &state_dict,
                                           float memory_limit_fraction) {
    const size_t physical_memory = get_apple_physical_memory_bytes();
    const size_t usable_memory = physical_memory * memory_limit_fraction;
    spdlog::debug("Physical/Usable memory available: {}/{} GB", physical_memory / BYTES_PER_GB,
                  usable_memory / BYTES_PER_GB);

    // Constrain the maximum batch size to use about half physical memory for decode buffers,
    // with neural network GPU buffers and CPU buffers assumed to occupy a subset of the
    // remaining memory.  This generally constrains the batch size to use fewer than
    // the maximum GPU cores when running sup models on systems with a large GPU core
    // to system memory ratio.
    const auto chunk_size = model_config.basecaller.chunk_size();
    const auto out_chunk_size = chunk_size / model_config.stride;

    const auto decode_buffer_size_per_elem =
            static_cast<size_t>(out_chunk_size) *
            (static_cast<size_t>(model_config.outsize) +        // Scores
             static_cast<size_t>(m_states) * sizeof(int16_t) +  // Posts
             static_cast<size_t>(m_states) * sizeof(float));    // Back guides.
    spdlog::trace("decode_buffer_size_per_elem {}", decode_buffer_size_per_elem);
    const int max_batch_size = static_cast<int>(
            std::clamp(utils::pad_to(usable_memory / (2 * decode_buffer_size_per_elem),
                                     static_cast<size_t>(MTL_CORE_BATCH_SIZE)),
                       static_cast<size_t>(MTL_CORE_BATCH_SIZE),
                       static_cast<size_t>(MTL_CORE_BATCH_SIZE * get_mtl_device_core_count())));
    spdlog::trace("max_batch_size {}", max_batch_size);

    // Subject to the above memory constraint, impose a minimum batch size
    // that will use 1/4 of GPU cores for LSTM execution.
    const int min_batch_size =
            std::min(MTL_CORE_BATCH_SIZE * get_mtl_device_core_count() / 4, max_batch_size);
    spdlog::trace("min_batch_size {}", min_batch_size);

    std::set<int> test_batch_sizes{max_batch_size};

    // Add some batch sizes evenly distributed in between.
    const int kNumSmallerSizes = 16;
    const float test_size_increment = static_cast<float>(max_batch_size - min_batch_size) /
                                      static_cast<float>(kNumSmallerSizes);
    for (int i = 0; i <= kNumSmallerSizes; ++i) {
        const int test_batch_size =
                utils::pad_to(min_batch_size + static_cast<int>(i * test_size_increment),
                              static_cast<int>(MTL_CORE_BATCH_SIZE));
        test_batch_sizes.insert(test_batch_size);
    }

    // To speed up test runs, use a smaller chunk size.  This means we will not see
    // the true effect of memory thrashing, so we are relying on the memory limit
    // above to avoid that scenario.
    const int benchmark_chunk_size =
            std::min(chunk_size, model_config.stride_inner() * 300 / model_config.scale_factor());

    // Iterate through batch size candidates to find the most efficient one.
    int best_batch_size = -1;
    long long best_us_per_batch_element = std::numeric_limits<long long>::max();
    for (int batch_size : test_batch_sizes) {
        spdlog::debug("Trying batch size {}", batch_size);
        set_chunk_batch_size(model_config, state_dict, benchmark_chunk_size, batch_size);
        auto dummy_input = torch::empty(
                {batch_size, benchmark_chunk_size, model_config.num_features}, torch::kF16);
        const auto start_time = std::chrono::system_clock::now();
        auto *cb = m_model->forward_async(dummy_input, nullptr, 0, 0, m_scores_TNC);
        run_scan_kernels(cb, 0);
        const auto end_time = std::chrono::system_clock::now();
        const auto elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                        .count();
        const auto us_per_batch_element = elapsed_us / batch_size;
        spdlog::debug("Batch {} us Batch element {} us", elapsed_us, us_per_batch_element);
        if (us_per_batch_element < best_us_per_batch_element) {
            best_us_per_batch_element = us_per_batch_element;
            best_batch_size = batch_size;
        }
    }
    assert(best_batch_size >= MTL_CORE_BATCH_SIZE);
    assert(best_batch_size % MTL_CORE_BATCH_SIZE == 0);
    return best_batch_size;
}

bool MetalLSTMCaller::run_scan_kernels(MTL::CommandBuffer *const cb, int try_count) {
    // This stage is operating on the split outputs of the linear layer, so
    // the effective batch size is m_out_batch_size.
    std::vector<int32_t> scan_args_{m_out_chunk_size, m_out_batch_size, m_states};
    auto scan_args = create_vec_buffer(m_device.get(), scan_args_);
    name_mtl_object(scan_args, "scan_kernel_args");

    for (int i = 0; i < m_out_split; ++i) {
        // TODO: optimise grid size
        launch_kernel_no_wait(m_bwd_scan_cps.get(), cb,
                              {scan_args.get(), mtl_for_tensor(m_scores_TNC.at(i)),
                               mtl_for_tensor(m_bwd_NTC.at(i))},
                              {}, m_out_batch_size, m_states);

        launch_kernel_no_wait(m_fwd_scan_add_softmax_cps.get(), cb,
                              {scan_args.get(), mtl_for_tensor(m_scores_TNC.at(i)),
                               mtl_for_tensor(m_bwd_NTC.at(i)), mtl_for_tensor(m_posts_NTC.at(i))},
                              {}, m_out_batch_size, m_states);
    }
    return finishCommandBuffer("linear/scan/softmax", cb, try_count);
}

bool MetalLSTMCaller::call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) {
    std::lock_guard lock(inter_caller_mutex);

    // The linear layer should not execute until the previous batch has been decoded,
    // since the same buffers are used for successive batches' scores, fwd/bwd scans.
    MTL::CommandBuffer *const cb =
            m_model->forward_async(*task.input, m_decode_complete_event.get(),
                                   task.decode_complete_event_id - 1, try_count, m_scores_TNC);
    if (cb == nullptr) {
        return false;
    }

    return run_scan_kernels(cb, try_count);
}

DecodedData MetalLSTMCaller::decode(int chunk_idx) const {
    // Model outputs are split across m_out_split buffers.
    assert(m_scores_TNC.size() == static_cast<size_t>(m_out_split));
    assert(m_bwd_NTC.size() == static_cast<size_t>(m_out_split));
    assert(m_posts_NTC.size() == static_cast<size_t>(m_out_split));
    const int out_buf_idx = chunk_idx / m_out_batch_size;
    const int buf_chunk_idx = chunk_idx % m_out_batch_size;

    return decode::beam_search_decode(
            // LSTM: m_scores_TNC[:, buf_chunk_idx, :] -> scores_TC
            m_scores_TNC.at(out_buf_idx).index({Slice(), buf_chunk_idx}),
            m_bwd_NTC.at(out_buf_idx)[buf_chunk_idx], m_posts_NTC.at(out_buf_idx)[buf_chunk_idx],
            m_decoder_options.beam_width, m_decoder_options.beam_cut, m_decoder_options.blank_score,
            m_decoder_options.q_shift, m_decoder_options.q_scale, m_score_scale);
}

MetalTxCaller::MetalTxCaller(const CRFModelConfig &model_config) : MetalCaller(model_config) {
    ScopedAutoReleasePool autorelease_pool;

    if (!model_config.is_tx_model()) {
        throw std::logic_error("MetalTxCaller got invalid model config");
    }

    m_device = get_mtl_device();
    m_command_queue = NS::TransferPtr(m_device->newCommandQueue());
    m_decode_complete_event = NS::TransferPtr(m_device->newSharedEvent());

    m_bwd_scan_float_cps = make_cps(m_device.get(), "backward_scan_float", {}, std::nullopt);
    m_fwd_scan_add_softmax_float_cps =
            make_cps(m_device.get(), "forward_scan_add_softmax_float", {}, std::nullopt);

    // Our metal builds assume shared memory, so it's safe to check host.
    if (auto total_mem = utils::total_host_memory_GB(); total_mem < 16) {
        spdlog::warn(
                "Less than 16GB of memory available: {}GB detected. "
                "This is below minimum spec and may cause issues",
                total_mem);
    }

    m_states = pow(m_config.tx->crf.n_base, model_config.state_len);

    m_decoder_options = decode::DecoderOptions();
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;

    if (m_decoder_options.blank_score != m_config.tx->crf.blank_score) {
        spdlog::warn("Transformer model config does not have the expected blank score");
    }

    assert(model_config.has_normalised_basecaller_params());
    m_in_chunk_size = model_config.basecaller.chunk_size();
    // Chunk size after decimation via convolution stride.
    m_out_chunk_size = m_in_chunk_size / model_config.stride;
    m_batch_size = model_config.basecaller.batch_size();

    assert(m_out_chunk_size > 0);
    assert(m_batch_size > 0);
    assert(model_config.outsize > 0);

    const int T = m_out_chunk_size;
    const int C = model_config.outsize;
    const int Cs = m_states;
    const int N = m_batch_size;

    m_scores_TNC = torch::empty({T, N, C}, m_scores_dtype);
    m_posts_NTC = torch::empty({N, T + 1, Cs}, m_posts_dtype);
    m_bwd_NTC = torch::empty({N, T + 1, Cs});

    load_tx_model(model_config);
    start_threads();
}

void MetalTxCaller::load_tx_model(const CRFModelConfig &model_config) {
    const auto device_type = torch::kMPS;
    const auto scalar_type = torch::kFloat16;
    const auto options = at::TensorOptions().device(device_type).dtype(scalar_type);

    m_model = nn::TxModel(model_config, options);

    auto state_dict = load_crf_model_weights(model_config);
    m_model->load_state_dict(state_dict);
    m_model->to(options.dtype().toScalarType());
    m_model->to(options.device());
    m_model->eval();
}

bool MetalTxCaller::run_scan_kernels(MTL::CommandBuffer *const cb, int try_count) {
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

    return finishCommandBuffer("linear/scan/softmax", cb, try_count);
}

bool MetalTxCaller::call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) {
    auto scores_TNC = m_model->forward(task.input->to(m_model->m_options))
                              .transpose(0, 1)
                              .contiguous()
                              .to(m_scores_dtype);

    MTL::CommandBuffer *const cb = next_command_buffer(m_command_queue.get(), try_count);
    if (m_decode_complete_event) {
        // wait for the previous decode task to complete - this acts as a mutex
        // previous scores are processed in the decode threads
        cb->encodeWait(m_decode_complete_event.get(), task.decode_complete_event_id - 1);
    }

    m_scores_TNC.index_put_({at::indexing::Ellipsis}, scores_TNC);

    std::lock_guard lock(inter_caller_mutex);
    return run_scan_kernels(cb, try_count);
}

DecodedData MetalTxCaller::decode(int chunk_idx) const {
    // Not splitting batches in Tx impl so chunk idx should be in [0, N)
    assert(chunk_idx < m_batch_size);
    return decode::beam_search_decode(
            m_scores_TNC.index({Slice(), chunk_idx}), m_bwd_NTC[chunk_idx], m_posts_NTC[chunk_idx],
            m_decoder_options.beam_width, m_decoder_options.beam_cut, m_decoder_options.blank_score,
            m_decoder_options.q_shift, m_decoder_options.q_scale, 1.0f);
};

}  // namespace dorado::basecall
