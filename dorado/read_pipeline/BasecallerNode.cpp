#include "BasecallerNode.h"

#include "basecall/CRFModelConfig.h"
#include "basecall/ModelRunnerBase.h"
#include "models/kits.h"
#include "read_utils.h"
#include "stitch.h"
#include "utils/stats.h"
#include "utils/thread_naming.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>

#if DORADO_METAL_BUILD
#include "torch_utils/metal_utils.h"
#endif

using namespace std::chrono_literals;
using namespace at::indexing;

namespace dorado {

struct BasecallerNode::BasecallingChunk : utils::Chunk {
    BasecallingChunk(std::shared_ptr<BasecallingRead> owner,
                     size_t offset,
                     size_t chunk_in_read_idx,
                     size_t chunk_size)
            : Chunk(offset, chunk_size),
              owning_read(std::move(owner)),
              idx_in_read(chunk_in_read_idx) {}

    std::shared_ptr<BasecallingRead> owning_read;  // The object that owns us.
    size_t idx_in_read;  // Just for tracking that the chunks don't go out of order.
};

struct BasecallerNode::BasecallingRead {
    Message read;                                              // The read itself.
    std::vector<std::unique_ptr<utils::Chunk>> called_chunks;  // Vector of basecalled chunks.
    std::atomic_size_t num_chunks_called;  // Number of chunks which have been basecalled.
};

size_t BasecallerNode::get_chunk_queue_idx(size_t read_raw_size) {
    // A read goes either to the queue with the smallest chunk size which can fit the whole read,
    // or, if the read is larger than all chunk sizes, the queue with the largest chunk size.
    size_t best_idx = 0;
    for (size_t i = 1; i < m_chunk_sizes.size(); ++i) {
        size_t best_size = m_chunk_sizes[best_idx];
        size_t this_size = m_chunk_sizes[i];
        if ((best_size < read_raw_size && best_size < this_size) ||
            (read_raw_size < this_size && this_size < best_size)) {
            best_idx = i;
        }
    }
    return best_idx;
}

void BasecallerNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.

        if (!is_read_message(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // Get the common read data.
        ReadCommon &read_common_data = get_read_common_data(message);
        // If a read has already been basecalled, just send it to the sink without basecalling again
        // TODO: This is necessary because some reads (e.g failed Stereo Encoding) will be passed
        // to the basecaller node having already been called. This should be fixed in the future with
        // support for graphs of nodes rather than linear pipelines.
        if (!read_common_data.seq.empty()) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this is a duplex read, raw_data won't have been generated yet.
        materialise_read_raw_data(message);

        // Now that we have acquired a read, wait until we can push to chunks_in
        // Chunk up the read and put the chunks into the pending chunk list.
        size_t raw_size = read_common_data.raw_data.sizes().back();  // Time dimension.
        size_t chunk_queue_idx = get_chunk_queue_idx(raw_size);
        size_t chunk_size = m_chunk_sizes[chunk_queue_idx];

        size_t offset = 0;
        size_t chunk_in_read_idx = 0;
        size_t signal_chunk_step = chunk_size - m_overlap;
        auto working_read = std::make_shared<BasecallingRead>();
        std::vector<std::unique_ptr<BasecallingChunk>> read_chunks;
        read_chunks.emplace_back(std::make_unique<BasecallingChunk>(
                working_read, offset, chunk_in_read_idx++, chunk_size));
        size_t num_chunks = 1;
        auto last_chunk_offset = raw_size > chunk_size ? raw_size - chunk_size : 0;
        auto misalignment = last_chunk_offset % m_model_stride;
        if (misalignment != 0) {
            // move last chunk start to the next stride boundary. we'll zero pad any excess samples required.
            last_chunk_offset += m_model_stride - misalignment;
        }
        while (offset + chunk_size < raw_size) {
            offset = std::min(offset + signal_chunk_step, last_chunk_offset);
            read_chunks.push_back(std::make_unique<BasecallingChunk>(
                    working_read, offset, chunk_in_read_idx++, chunk_size));
            ++num_chunks;
        }
        working_read->called_chunks.resize(num_chunks);
        working_read->num_chunks_called.store(0);
        working_read->read = std::move(message);

        // Put the read in the working list
        {
            std::lock_guard working_reads_lock(m_working_reads_mutex);
            m_working_reads_signal_bytes +=
                    get_read_common_data(working_read->read).raw_data.nbytes();
            m_working_reads.insert(std::move(working_read));
            ++m_working_reads_size;
        }

        // push the chunks to the chunk queue
        // needs to be done after working_read->read is set as chunks could be processed
        // before we set that value otherwise
        for (auto &chunk : read_chunks) {
            m_chunk_in_queues[chunk_queue_idx]->try_push(std::move(chunk));
        }
    }

    // Notify the basecaller threads that it is safe to gracefully terminate the basecaller
    for (auto &chunk_queue : m_chunk_in_queues) {
        chunk_queue->terminate();
    }
}

void BasecallerNode::basecall_current_batch(int worker_id) {
    NVTX3_FUNC_RANGE();
    auto &model_runner = m_model_runners[worker_id];
    auto &batched_chunks = m_batched_chunks[worker_id];
    spdlog::trace("Basecalling batch T={}, N={}, chunks={}, worker={}", model_runner->chunk_size(),
                  model_runner->batch_size(), batched_chunks.size(), worker_id);

    dorado::stats::Timer timer;
    auto decode_results = model_runner->call_chunks(int(batched_chunks.size()));
    m_call_chunks_ms += timer.GetElapsedMS();

    for (size_t i = 0; i < batched_chunks.size(); i++) {
        batched_chunks[i]->seq = std::move(decode_results[i].sequence);
        batched_chunks[i]->qstring = std::move(decode_results[i].qstring);
        batched_chunks[i]->moves = std::move(decode_results[i].moves);
    }

    for (auto &complete_chunk : batched_chunks) {
        m_processed_chunks.try_push(std::move(complete_chunk));
    }

    m_num_samples_incl_padding += model_runner->chunk_size() * model_runner->batch_size();
    if (batched_chunks.size() == model_runner->batch_size()) {
        ++m_num_batches_called;
    } else {
        ++m_num_partial_batches_called;
    }

    batched_chunks.clear();
}

void BasecallerNode::working_reads_manager() {
    utils::set_thread_name("bscl_reads_mgr");
    at::InferenceMode inference_mode_guard;

    std::unique_ptr<BasecallingChunk> chunk;
    while (m_processed_chunks.try_pop(chunk) == utils::AsyncQueueStatus::Success) {
        nvtx3::scoped_range loop{"working_reads_manager"};

        auto working_read = chunk->owning_read;
        auto idx_in_read = chunk->idx_in_read;
        working_read->called_chunks[idx_in_read] = std::move(chunk);
        auto num_chunks_called = ++working_read->num_chunks_called;
        if (num_chunks_called == working_read->called_chunks.size()) {
            // Finalise the read.
            auto source_read = std::move(working_read->read);

            ReadCommon &read_common_data = get_read_common_data(source_read);

            // model_stride is needed by the basecall server and stitch_chunks.
            read_common_data.model_stride = m_model_runners[0]->config().stride;

            // qbias/qscale are expected by the basecall server.
            read_common_data.model_q_bias = m_model_runners[0]->config().qbias;
            read_common_data.model_q_scale = m_model_runners[0]->config().qscale;

            utils::stitch_chunks(read_common_data, working_read->called_chunks);
            read_common_data.model_name = m_model_name;
            read_common_data.mean_qscore_start_pos = m_mean_qscore_start_pos;
            read_common_data.pre_trim_seq_length = read_common_data.seq.length();
            read_common_data.is_rna_model = m_is_rna_model;

            if (m_is_rna_model) {
                std::reverse(read_common_data.seq.begin(), read_common_data.seq.end());
                std::reverse(read_common_data.qstring.begin(), read_common_data.qstring.end());
            }

            // Update stats.
            ++m_called_reads_pushed;
            m_num_bases_processed += read_common_data.seq.length();
            m_num_samples_processed += read_common_data.get_raw_data_samples();

            // Chunks have ownership of the working read, so destroy them to avoid a leak.
            working_read->called_chunks.clear();

            // Do not trim R9.4.1 data to avoid changes to legacy products
            // Check here to avoid adding models lib as a dependency of utils
            if (read_common_data.chemistry != models::Chemistry::DNA_R9_4_1_E8) {
                // Trim reads which are affected by mux change and unblocking
                utils::mux_change_trim_read(read_common_data);
            }

            // Cleanup the working read.
            {
                std::unique_lock<std::mutex> working_reads_lock(m_working_reads_mutex);
                auto read_iter = m_working_reads.find(working_read);
                if (read_iter != m_working_reads.end()) {
                    m_working_reads_signal_bytes -= read_common_data.raw_data.nbytes();
                    m_working_reads.erase(read_iter);
                    --m_working_reads_size;
                } else {
                    throw std::runtime_error("Expected to find read id " +
                                             read_common_data.read_id +
                                             " in working reads cache but it doesn't exist.");
                }
            }

            // Send the read on its way.
            send_message_to_sink(std::move(source_read));
        }
    }
}

void BasecallerNode::basecall_worker_thread(int worker_id) {
    utils::set_thread_name("bscl_worker");
#if DORADO_METAL_BUILD
    // Model execution creates GPU-related autorelease objects.
    utils::ScopedAutoReleasePool outer_pool;
#endif
    at::InferenceMode inference_mode_guard;

    auto last_chunk_reserve_time = std::chrono::system_clock::now();
    const size_t batch_size = m_model_runners[worker_id]->batch_size();
    const size_t chunk_size = m_model_runners[worker_id]->chunk_size();
    const int batch_timeout_ms = m_model_runners[worker_id]->batch_timeout_ms();
    const int chunk_queue_idx = worker_id % int(m_chunk_in_queues.size());
    while (true) {
#if DORADO_METAL_BUILD
        utils::ScopedAutoReleasePool inner_pool;
#endif
        std::unique_ptr<BasecallingChunk> chunk;
        const auto pop_status = m_chunk_in_queues[chunk_queue_idx]->try_pop_until(
                chunk, last_chunk_reserve_time + std::chrono::milliseconds(batch_timeout_ms));

        if (pop_status == utils::AsyncQueueStatus::Terminate) {
            break;
        }

        if (pop_status == utils::AsyncQueueStatus::Timeout) {
            // try_pop_until timed out without getting a new chunk.
            if (!m_batched_chunks[worker_id].empty()) {
                // get scores for whatever chunks are available.
                basecall_current_batch(worker_id);
            }

            last_chunk_reserve_time = std::chrono::system_clock::now();
            continue;
        }

        // There's chunks to get_scores, so let's add them to our input tensor
        // FIXME -- it should not be possible to for this condition to be untrue.
        if (m_batched_chunks[worker_id].size() != batch_size) {
            // Copy the chunk into the input tensor
            auto &source_read = chunk->owning_read->read;

            auto &read_common = get_read_common_data(source_read);
            auto input_slice = read_common.raw_data.index(
                    {Ellipsis, Slice(chunk->input_offset, chunk->input_offset + chunk_size)});

            // Make sure the slice tensor is 2D
            if (input_slice.ndimension() == 1) {
                input_slice = input_slice.unsqueeze(0);
            }
            size_t slice_size = input_slice.size(1);

            // repeat-pad any non-full chunks
            if (slice_size != chunk_size) {
                auto [n, overhang] = std::div((int)chunk_size, (int)slice_size);
                input_slice = at::concat({input_slice.repeat({1, n}),
                                          input_slice.index({Ellipsis, Slice(0, overhang)})},
                                         1);
            }

            // Insert the chunk in the input tensor
            m_model_runners[worker_id]->accept_chunk(
                    static_cast<int>(m_batched_chunks[worker_id].size()), input_slice);

            m_batched_chunks[worker_id].push_back(std::move(chunk));

            last_chunk_reserve_time = std::chrono::system_clock::now();
        }

        if (m_batched_chunks[worker_id].size() == batch_size) {
            // Input tensor is full, let's get_scores.
            basecall_current_batch(worker_id);
        }
    }

    if (!m_batched_chunks[worker_id].empty()) {
        basecall_current_batch(worker_id);
    }

    // Reduce the count of active runner threads.  If this was the last active
    // thread also send termination signal to sink
    int num_remaining_runners = --m_num_active_model_runners;
    if (num_remaining_runners == 0) {
        // runners can share a caller, so shutdown when all runners are done
        // rather than terminating each runner as it finishes
        for (auto &runner : m_model_runners) {
            runner->terminate();
        }
        m_processed_chunks.terminate();
    }
}

namespace {

// Calculates the input queue size.
size_t CalcMaxChunksIn(const std::vector<basecall::RunnerPtr> &model_runners) {
    // Allow 2 batches per model runner on the chunks_in queue
    size_t max_chunks_in = 0;
    // Allows optimal batch size to be used for every GPU
    for (auto &runner : model_runners) {
        max_chunks_in += runner->batch_size() * 2;
    }
    return max_chunks_in;
}

}  // namespace

BasecallerNode::BasecallerNode(std::vector<basecall::RunnerPtr> model_runners,
                               size_t overlap,
                               std::string model_name,
                               size_t max_reads,
                               std::string node_name,
                               uint32_t read_mean_qscore_start_pos)
        : MessageSink(max_reads, 1),
          m_model_runners(std::move(model_runners)),
          m_overlap(overlap),
          m_model_stride(m_model_runners.front()->config().stride),
          m_is_rna_model(is_rna_model(m_model_runners.front()->config())),
          m_model_name(std::move(model_name)),
          m_mean_qscore_start_pos(read_mean_qscore_start_pos),
          m_processed_chunks(CalcMaxChunksIn(m_model_runners)),
          m_node_name(std::move(node_name)) {
    // Setup worker state
    const size_t num_workers = m_model_runners.size();
    m_batched_chunks.resize(num_workers);

    for (auto &runner_ptr : m_model_runners) {
        // m_model_runners is effectively a 3D array with dimensions
        // [num_devices][num_gpu_runners][num_chunk_sizes] (see
        // `dorado::api::create_basecall_runners`). This means the chunk sizes are repeated,
        // and to get the list of chunk sizes we iterate until we see the first chunk size again.
        if (!m_chunk_sizes.empty() && runner_ptr->chunk_size() == m_chunk_sizes[0]) {
            break;
        }
        m_chunk_sizes.push_back(runner_ptr->chunk_size());
    }

    auto chunk_queue_size = CalcMaxChunksIn(m_model_runners) / m_chunk_sizes.size();
    for (auto s : m_chunk_sizes) {
        m_chunk_in_queues.push_back(
                std::make_unique<utils::AsyncQueue<std::unique_ptr<BasecallingChunk>>>(
                        chunk_queue_size));
        spdlog::debug("BasecallerNode chunk size {}", s);
    }
}

BasecallerNode::~BasecallerNode() { terminate_impl(); }

void BasecallerNode::start_threads() {
    start_input_processing([this] { input_thread_fn(); }, "basecall_node");

    const size_t num_workers = m_model_runners.size();
    m_working_reads_managers.resize(std::max(size_t{1}, num_workers / 2));
    for (size_t i = 0; i < m_working_reads_managers.size(); i++) {
        m_working_reads_managers[i] = std::thread([this] { working_reads_manager(); });
    }
    m_basecall_workers.resize(num_workers);
    for (int i = 0; i < static_cast<int>(num_workers); i++) {
        m_basecall_workers[i] = std::thread([this, i] { basecall_worker_thread(i); });
    }
    m_num_active_model_runners = int(num_workers);
}

void BasecallerNode::terminate_impl() {
    stop_input_processing();
    for (auto &t : m_basecall_workers) {
        t.join();
    }
    m_basecall_workers.clear();
    for (auto &t : m_working_reads_managers) {
        t.join();
    }
    m_working_reads_managers.clear();
}

void BasecallerNode::restart() {
    for (auto &runner : m_model_runners) {
        runner->restart();
    }
    for (auto &chunk_queue : m_chunk_in_queues) {
        chunk_queue->restart();
    }
    m_processed_chunks.restart();
    start_threads();
}

stats::NamedStats BasecallerNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    for (const auto &runner : m_model_runners) {
        const auto runner_stats = stats::from_obj(*runner);
        stats.insert(runner_stats.begin(), runner_stats.end());
    }
    stats["batches_called"] = double(m_num_batches_called);
    stats["partial_batches_called"] = double(m_num_partial_batches_called);
    stats["call_chunks_ms"] = double(m_call_chunks_ms);
    stats["called_reads_pushed"] = double(m_called_reads_pushed);
    stats["working_reads_items"] = double(m_working_reads_size);
    stats["working_reads_signal_mb"] = double(m_working_reads_signal_bytes) / double((1024 * 1024));
    stats["bases_processed"] = double(m_num_bases_processed);
    stats["samples_processed"] = double(m_num_samples_processed);
    stats["samples_incl_padding"] = double(m_num_samples_incl_padding);
    return stats;
}

}  // namespace dorado
