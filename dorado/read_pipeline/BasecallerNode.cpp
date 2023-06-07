#include "BasecallerNode.h"

#include "../decode/CPUDecoder.h"
#include "utils/stats.h"
#include "utils/stitch.h"

#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <cstdlib>
#include <memory>

using namespace std::chrono_literals;
using namespace torch::indexing;

namespace dorado {

void BasecallerNode::input_worker_thread() {
    Message message;

    // Allow 5 batches per model runner on the chunks_in queue
    size_t max_chunks_in = 0;
    // Allows optimal batch size to be used for every GPU
    for (auto &runner : m_model_runners) {
        max_chunks_in += runner->batch_size() * 5;
    }

    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);
        // If a read has already been basecalled, just send it to the sink without basecalling again
        // TODO: This is necessary because some reads (e.g failed Stereo Encoding) will be passed
        // to the basecaller node having already been called. This should be fixed in the future with
        // support for graphs of nodes rather than linear pipelines.
        if (!read->seq.empty()) {
            m_sink.push_message(read);
            continue;
        }
        // Now that we have acquired a read, wait until we can push to chunks_in
        while (true) {
            std::unique_lock<std::mutex> chunk_lock(m_chunks_in_mutex);
            // A new condition was added to the condition variable which adjusts the predicate
            // to check for number of working reads. This is to deal with some degenerate cases during
            // duplex basecalling wherein the due to GPU contention between simplex and duplex stages
            // reads are not fully called, leading to a build up of working reads. These partial
            // reads were causing a growth in memory which sometimes led to a crash on some systems.
            // This change below more effectively puts a ceiling on the host memory usage.
            // Keeping the condition a function of the current sink size (empmirically at 5k reads this
            // caps memory around 30GB).
            m_chunks_in_has_space_cv.wait_for(chunk_lock, 10ms, [this, &max_chunks_in] {
                return (m_chunks_in.size() < max_chunks_in) &&
                       (m_working_reads.size() < 5 * m_max_reads);
            });

            if (m_chunks_in.size() >= max_chunks_in) {
                continue;
            }

            // Chunk up the read and put the chunks into the pending chunk list.
            size_t raw_size =
                    read->raw_data.sizes()[read->raw_data.sizes().size() - 1];  // Time dimension.

            size_t offset = 0;
            size_t chunk_in_read_idx = 0;
            size_t signal_chunk_step = m_chunk_size - m_overlap;
            m_chunks_in.push_back(
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
                m_chunks_in.push_back(
                        std::make_shared<Chunk>(read, offset, chunk_in_read_idx++, m_chunk_size));
                read->num_chunks++;
            }
            read->called_chunks.resize(read->num_chunks);
            read->num_chunks_called.store(0);
            chunk_lock.unlock();

            // Put the read in the working list
            {
                std::lock_guard working_reads_lock(m_working_reads_mutex);
                m_working_reads.push_back(std::move(read));
                ++m_working_reads_size;
            }

            m_chunks_added_cv.notify_one();

            break;  // Go back to watching the input reads
        }
    }

    // Notify the basecaller threads that it is safe to gracefully terminate the basecaller
    m_terminate_basecaller.store(true);
    m_chunks_added_cv.notify_all();
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

    // We need to assign each chunk back to the read it came from
    for (auto &complete_chunk : m_batched_chunks[worker_id]) {
        std::shared_ptr<Read> source_read = complete_chunk->source_read.lock();
        source_read->called_chunks[complete_chunk->idx_in_read] = complete_chunk;
        ++source_read->num_chunks_called;
    }
    m_batched_chunks[worker_id].clear();
    ++m_num_batches_called;
}

void BasecallerNode::working_reads_manager() {
    while (true) {
        nvtx3::scoped_range loop{"working_reads_manager"};

        std::deque<std::shared_ptr<Read>> completed_reads;
        {
            std::lock_guard working_reads_lock(m_working_reads_mutex);
            if (m_terminate_manager.load() && m_working_reads.empty()) {
                break;
            }
            for (auto read_iter = m_working_reads.begin(); read_iter != m_working_reads.end();) {
                if ((*read_iter)->num_chunks_called.load() == (*read_iter)->num_chunks) {
                    (*read_iter)->model_name =
                            m_model_name;  // Before sending read to sink, assign its model name
                    completed_reads.push_back(*read_iter);
                    read_iter = m_working_reads.erase(read_iter);
                    --m_working_reads_size;
                } else {
                    read_iter++;
                }
            }
        }

        if (completed_reads.empty()) {
            std::this_thread::sleep_for(10ms);
        } else {
            m_chunks_in_has_space_cv.notify_one();
        }

        for (auto &read : completed_reads) {
            utils::stitch_chunks(read);
            m_sink.push_message(read);
            ++m_called_reads_pushed;
            m_num_bases_processed += read->seq.length();
            m_num_samples_processed += read->raw_data.size(0);
        }
    }

    m_sink.terminate();
}

void BasecallerNode::basecall_worker_thread(int worker_id) {
    auto last_chunk_reserve_time = std::chrono::system_clock::now();
    int batch_size = m_model_runners[worker_id]->batch_size();
    while (true) {
        std::unique_lock<std::mutex> chunks_lock(m_chunks_in_mutex);
        if (!m_chunks_added_cv.wait_until(
                    chunks_lock,
                    last_chunk_reserve_time + std::chrono::milliseconds(m_batch_timeout_ms),
                    [this] { return !m_chunks_in.empty() || m_terminate_basecaller.load(); })) {
            // timeout without new chunks or termination call
            chunks_lock.unlock();
            if (!m_batched_chunks[worker_id].empty()) {
                basecall_current_batch(worker_id);
            }

            // reset wait period
            last_chunk_reserve_time = std::chrono::system_clock::now();
            continue;
        }

        if (m_chunks_in.empty() && m_terminate_basecaller.load()) {
            // no remaining chunks and we've been told to terminate
            // call the remaining batch
            chunks_lock.unlock();  // Not strictly necessary
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
                m_terminate_manager.store(true);
            }
            return;
        }

        // There's chunks to get_scores, so let's add them to our input tensor
        while (m_batched_chunks[worker_id].size() != batch_size && !m_chunks_in.empty()) {
            std::shared_ptr<Chunk> chunk = m_chunks_in.front();
            m_chunks_in.pop_front();
            chunks_lock.unlock();
            m_chunks_in_has_space_cv.notify_one();

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
            chunks_lock.lock();

            last_chunk_reserve_time = std::chrono::system_clock::now();
        }

        chunks_lock.unlock();

        if (m_batched_chunks[worker_id].size() == batch_size) {
            // Input tensor is full, let's get_scores.
            basecall_current_batch(worker_id);
        }
    }
}

BasecallerNode::BasecallerNode(MessageSink &sink,
                               std::vector<Runner> model_runners,
                               size_t overlap,
                               int batch_timeout_ms,
                               std::string model_name,
                               size_t max_reads)
        : MessageSink(max_reads),
          m_sink(sink),
          m_model_runners(std::move(model_runners)),
          m_chunk_size(m_model_runners.front()->chunk_size()),
          m_overlap(overlap),
          m_model_stride(m_model_runners.front()->model_stride()),
          m_terminate_basecaller(false),
          m_batch_timeout_ms(batch_timeout_ms),
          m_model_name(std::move(model_name)),
          m_max_reads(max_reads) {
    // Setup worker state
    size_t const num_workers = m_model_runners.size();
    m_batched_chunks.resize(num_workers);
    m_basecall_workers.resize(num_workers);
    m_num_active_model_runners = num_workers;

    initialization_time = std::chrono::system_clock::now();

    // Spin up any workers last so that we're not mutating |this| underneath them
    m_working_reads_manager = std::make_unique<std::thread>([this] { working_reads_manager(); });
    m_input_worker = std::make_unique<std::thread>([this] { input_worker_thread(); });
    for (int i = 0; i < static_cast<int>(num_workers); i++) {
        m_basecall_workers[i] = std::thread([this, i] { basecall_worker_thread(i); });
    }
}

BasecallerNode::~BasecallerNode() {
    terminate();
    m_input_worker->join();
    for (auto &t : m_basecall_workers) {
        t.join();
    }
    m_working_reads_manager->join();
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
