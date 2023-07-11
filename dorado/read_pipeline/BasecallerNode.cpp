#include "BasecallerNode.h"

#include "../decode/CPUDecoder.h"
#include "utils/stats.h"
#include "utils/stitch.h"

#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>

#if defined(__APPLE__) && !defined(__x86_64__)
#include "utils/metal_utils.h"
#endif

using namespace std::chrono_literals;
using namespace torch::indexing;

namespace dorado {

void BasecallerNode::input_worker_thread() {
    Message message;

    while (m_work_queue.try_pop(message)) {
        if (std::holds_alternative<CandidatePairRejectedMessage>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);
        // If a read has already been basecalled, just send it to the sink without basecalling again
        // TODO: This is necessary because some reads (e.g failed Stereo Encoding) will be passed
        // to the basecaller node having already been called. This should be fixed in the future with
        // support for graphs of nodes rather than linear pipelines.
        if (!read->seq.empty()) {
            send_message_to_sink(std::move(read));
            continue;
        }
        // Now that we have acquired a read, wait until we can push to chunks_in
        while (true) {
            // Chunk up the read and put the chunks into the pending chunk list.
            size_t raw_size =
                    read->raw_data.sizes()[read->raw_data.sizes().size() - 1];  // Time dimension.

            size_t offset = 0;
            size_t chunk_in_read_idx = 0;
            size_t signal_chunk_step = m_chunk_size - m_overlap;
            std::vector<std::shared_ptr<Chunk>> read_chunks;
            read_chunks.push_back(
                    std::make_shared<Chunk>(read, offset, chunk_in_read_idx++, m_chunk_size));
            read->num_chunks = 1;
            auto last_chunk_offset = raw_size - m_chunk_size;
            auto misalignment = last_chunk_offset % m_model_stride;
            if (misalignment != 0) {
                // move last chunk start to the next stride boundary. we'll zero pad any excess samples required.
                last_chunk_offset += m_model_stride - misalignment;
            }
            while (offset + m_chunk_size < raw_size) {
                offset = std::min(offset + signal_chunk_step, last_chunk_offset);
                read_chunks.push_back(
                        std::make_shared<Chunk>(read, offset, chunk_in_read_idx++, m_chunk_size));
                read->num_chunks++;
            }
            read->called_chunks.resize(read->num_chunks);
            read->num_chunks_called.store(0);

            // Put the read in the working list
            {
                std::lock_guard working_reads_lock(m_working_reads_mutex);
                m_working_reads.insert(std::move(read));
                ++m_working_reads_size;
            }

            for (auto &chunk : read_chunks) {
                m_chunks_in.try_push(std::move(chunk));
            }

            break;  // Go back to watching the input reads
        }
    }

    // Notify the basecaller threads that it is safe to gracefully terminate the basecaller
    m_chunks_in.terminate();
}

void BasecallerNode::basecall_current_batch(int worker_id) {
    NVTX3_FUNC_RANGE();
    auto model_runner = m_model_runners[worker_id];
    dorado::stats::Timer timer;
    auto decode_results = model_runner->call_chunks(m_batched_chunks[worker_id].size());
    m_call_chunks_ms += timer.GetElapsedMS();

    for (size_t i = 0; i < m_batched_chunks[worker_id].size(); i++) {
        m_batched_chunks[worker_id][i]->seq = decode_results[i].sequence;
        m_batched_chunks[worker_id][i]->qstring = decode_results[i].qstring;
        m_batched_chunks[worker_id][i]->moves = decode_results[i].moves;
    }

    for (auto &complete_chunk : m_batched_chunks[worker_id]) {
        m_processed_chunks.try_push(std::move(complete_chunk));
    }

    m_batched_chunks[worker_id].clear();
    ++m_num_batches_called;
}

void BasecallerNode::working_reads_manager() {
    std::shared_ptr<Chunk> chunk;
    while (m_processed_chunks.try_pop(chunk)) {
        nvtx3::scoped_range loop{"working_reads_manager"};

        auto source_read = chunk->source_read.lock();
        source_read->called_chunks[chunk->idx_in_read] = chunk;
        auto num_chunks_called = ++source_read->num_chunks_called;
        if (num_chunks_called == source_read->num_chunks) {
            utils::stitch_chunks(source_read);
            ++m_called_reads_pushed;
            m_num_bases_processed += source_read->seq.length();
            m_num_samples_processed += source_read->raw_data.size(0);

            std::shared_ptr<Read> found_read;
            {
                std::unique_lock<std::mutex> working_reads_lock(m_working_reads_mutex);
                auto read_iter = m_working_reads.find(source_read);
                if (read_iter != m_working_reads.end()) {
                    (*read_iter)->model_name = m_model_name;
                    (*read_iter)->mean_qscore_start_pos = m_mean_qscore_start_pos;
                    found_read = std::move(*read_iter);
                    m_working_reads.erase(read_iter);
                    --m_working_reads_size;
                } else {
                    throw std::runtime_error("Expected to find read id " + source_read->read_id +
                                             " in working reads cache but it doesn't exist.");
                }
            }
            send_message_to_sink(std::move(found_read));
        }
    }
}

void BasecallerNode::basecall_worker_thread(int worker_id) {
#if defined(__APPLE__) && !defined(__x86_64__)
    // Model execution creates GPU-related autorelease objects.
    utils::ScopedAutoReleasePool autorelease_pool;
#endif
    auto last_chunk_reserve_time = std::chrono::system_clock::now();
    int batch_size = m_model_runners[worker_id]->batch_size();
    std::shared_ptr<Chunk> chunk;
    while (m_chunks_in.try_pop_until(
            chunk, last_chunk_reserve_time + std::chrono::milliseconds(m_batch_timeout_ms))) {
        // If chunk is empty, then try_pop timed out without getting a new chunk.
        if (!chunk) {
            if (!m_batched_chunks[worker_id].empty()) {
                // get scores for whatever chunks are available.
                basecall_current_batch(worker_id);
            }

            last_chunk_reserve_time = std::chrono::system_clock::now();
            continue;
        }

        // There's chunks to get_scores, so let's add them to our input tensor
        if (m_batched_chunks[worker_id].size() != batch_size) {
            // Copy the chunk into the input tensor
            std::shared_ptr<Read> source_read = chunk->source_read.lock();

            auto input_slice = source_read->raw_data.index(
                    {Ellipsis, Slice(chunk->input_offset, chunk->input_offset + m_chunk_size)});
            size_t slice_size;
            if (input_slice.ndimension() == 1) {
                slice_size = input_slice.size(0);
            } else {
                slice_size = input_slice.sizes()[1];
            }

            // repeat-pad any non-full chunks
            // Stereo and Simplex encoding need to be treated differently
            if (slice_size != m_chunk_size) {
                if (input_slice.ndimension() == 1) {
                    auto [n, overhang] = std::div((int)m_chunk_size, (int)slice_size);
                    input_slice = torch::concat(
                            {input_slice.repeat({n}),
                             input_slice.index({Ellipsis, torch::indexing::Slice(0, overhang)})});
                } else if (input_slice.ndimension() == 2) {
                    auto [n, overhang] = std::div((int)m_chunk_size, (int)slice_size);
                    input_slice = torch::concat(
                            {input_slice.repeat({1, n}),
                             input_slice.index({Ellipsis, torch::indexing::Slice(0, overhang)})},
                            1);
                }
            }

            // Insert the chunk in the input tensor
            m_model_runners[worker_id]->accept_chunk(
                    static_cast<int>(m_batched_chunks[worker_id].size()), input_slice);

            m_batched_chunks[worker_id].push_back(chunk);

            last_chunk_reserve_time = std::chrono::system_clock::now();
        }

        if (m_batched_chunks[worker_id].size() == batch_size) {
            // Input tensor is full, let's get_scores.
            basecall_current_batch(worker_id);
        }
        chunk.reset();
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
size_t CalcMaxChunksIn(const std::vector<Runner> &model_runners) {
    // Allow 5 batches per model runner on the chunks_in queue
    size_t max_chunks_in = 0;
    // Allows optimal batch size to be used for every GPU
    for (auto &runner : model_runners) {
        max_chunks_in += runner->batch_size() * 5;
    }
    return max_chunks_in;
}

}  // namespace

BasecallerNode::BasecallerNode(std::vector<Runner> model_runners,
                               size_t overlap,
                               int batch_timeout_ms,
                               std::string model_name,
                               size_t max_reads,
                               const std::string &node_name,
                               bool in_duplex_pipeline,
                               uint32_t read_mean_qscore_start_pos)
        : MessageSink(max_reads),
          m_model_runners(std::move(model_runners)),
          m_chunk_size(m_model_runners.front()->chunk_size()),
          m_overlap(overlap),
          m_model_stride(m_model_runners.front()->model_stride()),
          m_batch_timeout_ms(batch_timeout_ms),
          m_model_name(std::move(model_name)),
          m_max_reads(max_reads),
          m_in_duplex_pipeline(in_duplex_pipeline),
          m_mean_qscore_start_pos(read_mean_qscore_start_pos),
          m_chunks_in(CalcMaxChunksIn(m_model_runners)),
          m_processed_chunks(CalcMaxChunksIn(m_model_runners)),
          m_node_name(node_name) {
    // Setup worker state
    const size_t num_workers = m_model_runners.size();
    m_batched_chunks.resize(num_workers);
    m_basecall_workers.resize(num_workers);
    m_num_active_model_runners = num_workers;

    initialization_time = std::chrono::system_clock::now();

    // Spin up any workers last so that we're not mutating |this| underneath them
    m_working_reads_managers.resize(num_workers / 2);
    for (int i = 0; i < m_working_reads_managers.size(); i++) {
        m_working_reads_managers[i] = std::thread([this] { working_reads_manager(); });
    }
    m_input_worker = std::make_unique<std::thread>([this] { input_worker_thread(); });
    for (int i = 0; i < static_cast<int>(num_workers); i++) {
        m_basecall_workers[i] = std::thread([this, i] { basecall_worker_thread(i); });
    }
}

void BasecallerNode::terminate_impl() {
    terminate_input_queue();
    if (m_input_worker->joinable()) {
        m_input_worker->join();
    }
    for (auto &t : m_basecall_workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    for (auto &t : m_working_reads_managers) {
        if (t.joinable()) {
            t.join();
        }
    }
    termination_time = std::chrono::system_clock::now();
}

stats::NamedStats BasecallerNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    for (const auto &runner : m_model_runners) {
        const auto runner_stats = stats::from_obj(*runner);
        stats.insert(runner_stats.begin(), runner_stats.end());
    }
    stats["batches_called"] = m_num_batches_called;
    stats["partial_batches_called"] = m_num_partial_batches_called;
    stats["call_chunks_ms"] = m_call_chunks_ms;
    stats["called_reads_pushed"] = m_called_reads_pushed;
    stats["working_reads_items"] = m_working_reads_size;
    stats["bases_processed"] = m_num_bases_processed;
    stats["samples_processed"] = m_num_samples_processed;
    return stats;
}

}  // namespace dorado
