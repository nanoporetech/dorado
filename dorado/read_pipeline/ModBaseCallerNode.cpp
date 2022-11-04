#include "ModBaseCallerNode.h"

#include "modbase/remora_encoder.h"
#include "modbase/remora_utils.h"
#include "nn/RemoraModel.h"
#include "utils/base_mod_utils.h"
#include "utils/math_utils.h"
#include "utils/sequence_utils.h"

#include <chrono>
using namespace std::chrono_literals;

namespace dorado {

constexpr auto FORCE_TIMEOUT = 100ms;

ModBaseCallerNode::ModBaseCallerNode(ReadSink& sink,
                                     std::vector<std::shared_ptr<RemoraCaller>> model_callers,
                                     size_t remora_threads,
                                     size_t num_devices,
                                     size_t block_stride,
                                     size_t batch_size,
                                     size_t max_reads)
        : ReadSink(max_reads),
          m_sink(sink),
          m_num_devices(num_devices),
          m_batch_size(batch_size),
          m_block_stride(block_stride),
          m_callers(std::move(model_callers)) {
    init_modbase_info();

    m_output_worker = std::make_unique<std::thread>(&ModBaseCallerNode::output_worker_thread, this);

    size_t num_model_callers = m_callers.size();

    m_chunk_queues.resize(num_model_callers / num_devices);
    m_batched_chunks.resize(num_model_callers);

    for (size_t i = 0; i < num_model_callers; i++) {
        std::unique_ptr<std::thread> t =
                std::make_unique<std::thread>(&ModBaseCallerNode::caller_worker_thread, this, i);
        m_caller_workers.push_back(std::move(t));
        ++m_num_active_model_callers;
    }

    // Spin up the proessing threads:
    for (size_t i = 0; i < remora_threads * num_devices; ++i) {
        std::unique_ptr<std::thread> t =
                std::make_unique<std::thread>(&ModBaseCallerNode::runner_worker_thread, this, i);
        m_runner_workers.push_back(std::move(t));
        ++m_num_active_model_runners;
    }
}

ModBaseCallerNode::~ModBaseCallerNode() {
    terminate();
    m_cv.notify_all();
    for (auto& t : m_runner_workers) {
        t->join();
    }
    for (auto& t : m_caller_workers) {
        t->join();
    }
    m_output_worker->join();
}

void ModBaseCallerNode::init_modbase_info() {
    struct Info {
        std::vector<std::string> long_names;
        std::string alphabet;
        std::string motif;
        int motif_offset;
    };

    std::string allowed_bases = "ACGT";
    std::array<Info, 4> model_info;
    for (int b = 0; b < 4; ++b) {
        model_info[b].alphabet = allowed_bases[b];
    }

    std::array<size_t, 4> base_counts = {1, 1, 1, 1};
    for (size_t id = 0; id < m_callers.size() / m_num_devices; ++id) {
        const auto& params = m_callers[id]->params();

        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            throw std::runtime_error("Invalid base in remora model metadata.");
        }
        auto& map_entry = model_info[RemoraUtils::BASE_IDS[base]];
        map_entry.long_names = params.mod_long_names;
        map_entry.alphabet += params.mod_bases;
        map_entry.motif = params.motif;
        map_entry.motif_offset = params.motif_offset;

        base_counts[RemoraUtils::BASE_IDS[base]] = params.base_mod_count + 1;
        m_num_states += params.base_mod_count;
    }

    std::string long_names, alphabet;
    utils::BaseModContext context_handler;
    for (const auto& info : model_info) {
        for (const auto& name : info.long_names) {
            if (!long_names.empty())
                long_names += ' ';
            long_names += name;
        }
        alphabet += info.alphabet;
        if (!info.motif.empty()) {
            context_handler.set_context(info.motif, size_t(info.motif_offset));
        }
    }

    m_base_mod_info =
            std::make_shared<utils::BaseModInfo>(alphabet, long_names, context_handler.encode());

    m_base_prob_offsets[0] = 0;
    m_base_prob_offsets[1] = base_counts[0];
    m_base_prob_offsets[2] = base_counts[0] + base_counts[1];
    m_base_prob_offsets[3] = base_counts[0] + base_counts[1] + base_counts[2];
}

void ModBaseCallerNode::runner_worker_thread(size_t runner_id) {
    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {
                int num_remaining_runners = --m_num_active_model_runners;
                if (num_remaining_runners == 0) {
                    m_terminate_callers = true;
                }
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        size_t max_chunks_in = m_batch_size * 5;  // size per queue: one queue per caller
        auto chunk_queues_available = [this, &max_chunks_in] {
            return std::all_of(
                    std::begin(m_chunk_queues), std::end(m_chunk_queues),
                    [&max_chunks_in](const auto& queue) { return queue.size() < max_chunks_in; });
        };

        while (true) {
            std::unique_lock<std::mutex> chunk_lock(m_chunk_queues_mutex);
            m_chunk_queues_cv.wait_for(chunk_lock, 10ms, chunk_queues_available);
            if (!chunk_queues_available()) {
                continue;
            }
            chunk_lock.unlock();

            // initialize base_mod_probs _before_ we start handing out chunks
            read->base_mod_probs.resize(read->seq.size() * m_num_states, 0);
            for (size_t i = 0; i < read->seq.size(); ++i) {
                // Initialize for what corresponds to 100% canonical base for each position.
                int base_id = RemoraUtils::BASE_IDS[read->seq[i]];
                if (base_id < 0) {
                    throw std::runtime_error("Invalid character in sequence.");
                }
                read->base_mod_probs[i * m_num_states + m_base_prob_offsets[base_id]] = 1.0f;
            }
            read->base_mod_info = m_base_mod_info;

            std::vector<int> sequence_ints = utils::sequence_to_ints(read->seq);
            std::vector<uint64_t> seq_to_sig_map = utils::moves_to_map(
                    read->moves, m_block_stride, read->raw_data.size(0), read->seq.size() + 1);

            read->num_modbase_chunks = 0;
            read->num_modbase_chunks_called = 0;
            for (size_t caller_id = 0; caller_id < m_callers.size() / m_num_devices; ++caller_id) {
                const auto& caller = m_callers[caller_id];
                auto& chunk_queue = m_chunk_queues[caller_id];

                // scale signal based on model parameters
                auto scaled_signal =
                        caller->scale_signal(read->raw_data, sequence_ints, seq_to_sig_map);

                auto& params = caller->params();
                auto context_samples = (params.context_before + params.context_after);
                // One-hot encodes the kmer at each signal step for input into the network
                RemoraEncoder encoder(m_block_stride, context_samples, params.bases_before,
                                      params.bases_after);
                encoder.init(sequence_ints, seq_to_sig_map);

                auto context_hits = caller->get_motif_hits(read->seq);
                for (auto context_hit : context_hits) {
                    auto slice = encoder.get_context(context_hit);
                    auto input_signal = scaled_signal.index({torch::indexing::Slice(
                            slice.first_sample, slice.first_sample + slice.num_samples)});
                    if (slice.lead_samples_needed != 0 || slice.tail_samples_needed != 0) {
                        input_signal = torch::constant_pad_nd(input_signal,
                                                              {(int64_t)slice.lead_samples_needed,
                                                               (int64_t)slice.tail_samples_needed});
                    }

                    chunk_lock.lock();
                    chunk_queue.push_back(std::make_shared<RemoraChunk>(
                            read, input_signal, std::move(slice.data), context_hit));
                    chunk_lock.unlock();

                    ++read->num_modbase_chunks;
                }
            }

            // Put the read in the working list
            std::unique_lock<std::mutex> working_reads_lock(m_working_reads_mutex);
            m_working_reads.push_back(read);
            working_reads_lock.unlock();
            break;
        }
    }
}

void ModBaseCallerNode::caller_worker_thread(size_t caller_id) {
    auto& caller = m_callers[caller_id];

    auto num_models = m_callers.size() / m_num_devices;
    auto& chunk_queue = m_chunk_queues[caller_id % num_models];
    auto& batched_chunks = m_batched_chunks[caller_id];
    auto last_chunk_reserve_time = std::chrono::system_clock::now();

    while (true) {
        std::unique_lock<std::mutex> chunks_lock(m_chunk_queues_mutex);
        if (chunk_queue.empty()) {
            if (m_terminate_callers) {
                chunks_lock.unlock();  // Not strictly necessary
                // We dispatch any part-full buffer here to finish basecalling.
                if (!batched_chunks.empty()) {
                    call_current_batch(caller_id);
                }

                //Reduce the count of active model callers, if this was the last active model caller also send termination signal to sink
                int num_remaining_callers = --m_num_active_model_callers;
                if (num_remaining_callers == 0) {
                    m_terminate_output = true;
                }
                return;
            } else {
                // There's no chunks available to call at the moment, sleep and try again
                chunks_lock.unlock();

                auto current_time = std::chrono::system_clock::now();
                auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                        current_time - last_chunk_reserve_time);
                if (delta > FORCE_TIMEOUT && !batched_chunks.empty()) {
                    call_current_batch(caller_id);
                } else {
                    std::this_thread::sleep_for(100ms);
                }
                continue;
            }
        }

        while (batched_chunks.size() != m_batch_size && !chunk_queue.empty()) {
            std::shared_ptr<RemoraChunk> chunk = chunk_queue.front();
            chunk_queue.pop_front();
            chunks_lock.unlock();

            caller->accept_chunk(batched_chunks.size(), chunk->signal, chunk->encoded_kmers);
            batched_chunks.push_back(chunk);
            chunks_lock.lock();

            last_chunk_reserve_time = std::chrono::system_clock::now();
        }

        chunks_lock.unlock();
        m_chunk_queues_cv.notify_one();
        if (m_batched_chunks[caller_id].size() == m_batch_size) {
            // Input tensor is full, let's get_scores.
            call_current_batch(caller_id);
        }
    }
}

void ModBaseCallerNode::call_current_batch(size_t caller_id) {
    auto& caller = m_callers[caller_id];
    auto results = caller->call_chunks(m_batched_chunks[caller_id].size());

    std::unique_lock processed_chunks_lock(m_processed_chunks_mutex);
    auto row_size = results.size(1);

    // Put results into chunk
    for (size_t i = 0; i < m_batched_chunks[caller_id].size(); ++i) {
        auto& chunk = m_batched_chunks[caller_id][i];
        chunk->scores.resize(row_size);
        for (int j = 0; j < row_size; ++j) {
            chunk->scores[j] = results.index({(int)i, j}).item().toFloat();
        }
        m_processed_chunks.push_back(chunk);
    }

    processed_chunks_lock.unlock();
    m_processed_chunks_cv.notify_one();

    std::unique_lock<std::mutex> chunks_lock(m_chunk_queues_mutex);
    m_batched_chunks[caller_id].clear();
    chunks_lock.unlock();
}

void ModBaseCallerNode::output_worker_thread() {
    while (true) {
        // Wait until we are provided with a read
        std::unique_lock processed_chunks_lock(m_processed_chunks_mutex);
        m_processed_chunks_cv.wait_for(processed_chunks_lock, 100ms,
                                       [this] { return !m_processed_chunks.empty(); });
        if (m_processed_chunks.empty()) {
            if (m_terminate_output) {
                m_sink.terminate();
                return;
            } else {
                continue;
            }
        }

        for (const auto& chunk : m_processed_chunks) {
            auto source_read = chunk->source_read.lock();
            int64_t result_pos = chunk->context_hit;
            int64_t offset =
                    m_base_prob_offsets[RemoraUtils::BASE_IDS[source_read->seq[result_pos]]];
            for (size_t i = 0; i < chunk->scores.size(); ++i) {
                source_read->base_mod_probs[m_num_states * result_pos + offset + i] =
                        uint8_t(std::min(std::floor(chunk->scores[i] * 256), 255.0f));
            }
            source_read->num_modbase_chunks_called += 1;
        }

        m_processed_chunks.clear();
        processed_chunks_lock.unlock();

        // Now move any completed reads to the output queue
        std::unique_lock<std::mutex> working_reads_lock(m_working_reads_mutex);
        for (auto read_iter = m_working_reads.begin(); read_iter != m_working_reads.end();) {
            if ((*read_iter)->num_modbase_chunks_called.load() ==
                (*read_iter)->num_modbase_chunks) {
                m_sink.push_read(*read_iter);
                read_iter = m_working_reads.erase(read_iter);
            } else {
                ++read_iter;
            }
        }
        working_reads_lock.unlock();
    }
}

}  // namespace dorado
