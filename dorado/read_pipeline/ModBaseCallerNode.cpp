#include "ModBaseCallerNode.h"

#include "modbase/remora_encoder.h"
#include "modbase/remora_utils.h"
#include "nn/ModBaseRunner.h"
#include "utils/base_mod_utils.h"
#include "utils/math_utils.h"
#include "utils/sequence_utils.h"
#include "utils/stats.h"
#include "utils/tensor_utils.h"

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstring>
using namespace std::chrono_literals;

namespace dorado {

constexpr auto FORCE_TIMEOUT = 100ms;

ModBaseCallerNode::ModBaseCallerNode(std::vector<std::unique_ptr<ModBaseRunner>> model_runners,
                                     size_t remora_threads,
                                     size_t block_stride,
                                     size_t batch_size,
                                     size_t max_reads)
        : MessageSink(max_reads),
          m_batch_size(batch_size),
          m_block_stride(block_stride),
          m_runners(std::move(model_runners)),
          // TODO -- more principled calculation of output queue size
          m_processed_chunks(10 * max_reads) {
    init_modbase_info();

    m_output_worker = std::make_unique<std::thread>(&ModBaseCallerNode::output_worker_thread, this);

    m_chunk_queues.resize(m_runners[0]->num_callers());

    for (size_t worker_id = 0; worker_id < m_runners.size(); ++worker_id) {
        for (size_t model_id = 0; model_id < m_runners[worker_id]->num_callers(); ++model_id) {
            std::unique_ptr<std::thread> t = std::make_unique<std::thread>(
                    &ModBaseCallerNode::modbasecall_worker_thread, this, worker_id, model_id);
            m_runner_workers.push_back(std::move(t));
            ++m_num_active_runner_workers;
        }
    }
    // Spin up the processing threads:
    for (size_t i = 0; i < remora_threads; ++i) {
        std::unique_ptr<std::thread> t =
                std::make_unique<std::thread>(&ModBaseCallerNode::input_worker_thread, this);
        m_input_worker.push_back(std::move(t));
        ++m_num_active_input_worker;
    }
}

void ModBaseCallerNode::terminate_impl() {
    terminate_input_queue();
    for (auto& t : m_input_worker) {
        if (t->joinable()) {
            t->join();
        }
    }
    for (auto& t : m_runner_workers) {
        if (t->joinable()) {
            t->join();
        }
    }
    if (m_output_worker->joinable()) {
        m_output_worker->join();
    }
}

[[maybe_unused]] ModBaseCallerNode::Info ModBaseCallerNode::get_modbase_info_and_maybe_init(
        std::vector<std::reference_wrapper<ModBaseParams const>> const& base_mod_params,
        ModBaseCallerNode* node) {
    struct ModelInfo {
        std::vector<std::string> long_names;
        std::string alphabet;
        std::string motif;
        int motif_offset;
        size_t base_counts = 1;
    };

    std::string const allowed_bases = "ACGT";
    std::array<ModelInfo, 4> model_info;
    for (int b = 0; b < 4; ++b) {
        model_info[b].alphabet = allowed_bases[b];
    }

    for (const auto& params_ref : base_mod_params) {
        const auto& params = params_ref.get();
        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            throw std::runtime_error("Invalid base in remora model metadata.");
        }
        auto& map_entry = model_info[RemoraUtils::BASE_IDS[base]];
        map_entry.long_names = params.mod_long_names;
        map_entry.alphabet += params.mod_bases;
        if (node) {
            map_entry.motif = params.motif;
            map_entry.motif_offset = params.motif_offset;
            map_entry.base_counts = params.base_mod_count + 1;
            node->m_num_states += params.base_mod_count;
        }
    }

    Info result;
    utils::BaseModContext context_handler;
    for (const auto& info : model_info) {
        for (const auto& name : info.long_names) {
            if (!result.long_names.empty())
                result.long_names += ' ';
            result.long_names += name;
        }
        result.alphabet += info.alphabet;
        if (node && !info.motif.empty()) {
            context_handler.set_context(info.motif, size_t(info.motif_offset));
        }
    }

    if (node) {
        node->m_base_mod_info = std::make_shared<utils::BaseModInfo>(
                result.alphabet, result.long_names, context_handler.encode());

        node->m_base_prob_offsets[0] = 0;
        node->m_base_prob_offsets[1] = model_info[0].base_counts;
        node->m_base_prob_offsets[2] = node->m_base_prob_offsets[1] + model_info[1].base_counts;
        node->m_base_prob_offsets[3] = node->m_base_prob_offsets[2] + model_info[2].base_counts;
    }

    return result;
}

void ModBaseCallerNode::init_modbase_info() {
    std::vector<std::reference_wrapper<ModBaseParams const>> base_mod_params;
    auto& runner = m_runners[0];
    for (size_t caller_id = 0; caller_id < runner->num_callers(); ++caller_id) {
        base_mod_params.emplace_back(runner->caller_params(caller_id));
    }
    get_modbase_info_and_maybe_init(base_mod_params, this);
}

void ModBaseCallerNode::input_worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        nvtx3::scoped_range range{"modbase_input_worker_thread"};
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        const size_t max_chunks_in = m_batch_size * 5;  // size per queue: one queue per caller
        auto chunk_queues_available = [this, &max_chunks_in] {
            return std::all_of(
                    std::begin(m_chunk_queues), std::end(m_chunk_queues),
                    [&max_chunks_in](const auto& queue) { return queue.size() < max_chunks_in; });
        };

        while (true) {
            std::unique_lock<std::mutex> chunk_lock(m_chunk_queues_mutex);
            m_chunk_queues_cv.wait(chunk_lock, chunk_queues_available);
            chunk_lock.unlock();

            stats::Timer timer;
            {
                nvtx3::scoped_range range{"base_mod_probs_init"};
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
            }
            read->base_mod_info = m_base_mod_info;

            std::vector<int> sequence_ints = utils::sequence_to_ints(read->seq);
            std::vector<uint64_t> seq_to_sig_map = utils::moves_to_map(
                    read->moves, m_block_stride, read->raw_data.size(0), read->seq.size() + 1);

            read->num_modbase_chunks = 0;
            read->num_modbase_chunks_called = 0;

            // all runners have the same set of callers, so we only need to use the first one
            auto& runner = m_runners[0];
            for (size_t caller_id = 0; caller_id < runner->num_callers(); ++caller_id) {
                nvtx3::scoped_range range{"generate_chunks"};
                auto& chunk_queue = m_chunk_queues[caller_id];

                // scale signal based on model parameters
                auto scaled_signal = runner->scale_signal(caller_id, read->raw_data, sequence_ints,
                                                          seq_to_sig_map);

                auto& params = runner->caller_params(caller_id);
                auto context_samples = (params.context_before + params.context_after);
                // One-hot encodes the kmer at each signal step for input into the network
                RemoraEncoder encoder(m_block_stride, context_samples, params.bases_before,
                                      params.bases_after);
                encoder.init(sequence_ints, seq_to_sig_map);

                auto context_hits = runner->get_motif_hits(caller_id, read->seq);
                m_num_context_hits += static_cast<int64_t>(context_hits.size());
                std::vector<std::shared_ptr<RemoraChunk>> reads_to_enqueue;
                reads_to_enqueue.reserve(context_hits.size());
                for (auto context_hit : context_hits) {
                    nvtx3::scoped_range range{"create_chunk"};
                    auto slice = encoder.get_context(context_hit);
                    // signal
                    auto input_signal = scaled_signal.index({torch::indexing::Slice(
                            slice.first_sample, slice.first_sample + slice.num_samples)});
                    if (slice.lead_samples_needed != 0 || slice.tail_samples_needed != 0) {
                        input_signal = torch::constant_pad_nd(input_signal,
                                                              {(int64_t)slice.lead_samples_needed,
                                                               (int64_t)slice.tail_samples_needed});
                    }
                    reads_to_enqueue.push_back(std::make_shared<RemoraChunk>(
                            read, input_signal, std::move(slice.data), context_hit));

                    ++read->num_modbase_chunks;
                }
                chunk_lock.lock();
                chunk_queue.insert(chunk_queue.end(), reads_to_enqueue.begin(),
                                   reads_to_enqueue.end());
                chunk_lock.unlock();
                reads_to_enqueue.size() > m_batch_size ? m_chunks_added_cv.notify_all()
                                                       : m_chunks_added_cv.notify_one();
            }
            m_chunk_generation_ms += timer.GetElapsedMS();

            if (read->num_modbase_chunks != 0) {
                // Put the read in the working list
                std::scoped_lock<std::mutex> working_reads_lock(m_working_reads_mutex);
                m_working_reads.push_back(read);
            } else {
                // No modbases to call, pass directly to next node
                send_message_to_sink(read);
                ++m_num_non_mod_base_reads_pushed;
            }
            break;
        }
    }

    int num_remaining_workers = --m_num_active_input_worker;
    if (num_remaining_workers == 0) {
        m_terminate_runners.store(true);
        m_chunks_added_cv.notify_all();
    }
}

void ModBaseCallerNode::modbasecall_worker_thread(size_t worker_id, size_t caller_id) {
    auto& runner = m_runners[worker_id];
    auto& chunk_queue = m_chunk_queues[caller_id];

    auto batched_chunks = std::vector<std::shared_ptr<RemoraChunk>>{};
    auto last_chunk_reserve_time = std::chrono::system_clock::now();

    while (true) {
        nvtx3::scoped_range range{"modbasecall_worker_thread"};
        std::unique_lock<std::mutex> chunks_lock(m_chunk_queues_mutex);
        if (!m_chunks_added_cv.wait_until(
                    chunks_lock, last_chunk_reserve_time + FORCE_TIMEOUT, [&chunk_queue, this] {
                        return !chunk_queue.empty() || m_terminate_runners.load();
                    })) {
            // timeout without new chunks or termination call
            chunks_lock.unlock();
            if (!batched_chunks.empty()) {
                call_current_batch(worker_id, caller_id, batched_chunks);
            }

            // reset wait period
            last_chunk_reserve_time = std::chrono::system_clock::now();
            continue;
        }

        if (chunk_queue.empty() && m_terminate_runners.load()) {
            // no remaining chunks and we've been told to terminate
            // call the remaining batch
            chunks_lock.unlock();  // Not strictly necessary
            if (!batched_chunks.empty()) {
                call_current_batch(worker_id, caller_id, batched_chunks);
            }

            // Reduce the count of active runner threads.  If this was the last active
            // thread also send termination signal to sink
            int num_remaining_runners = --m_num_active_runner_workers;
            if (num_remaining_runners == 0) {
                // runners can share a caller, so shutdown when all runners are done
                // rather than terminating each runner as it finishes
                for (auto& runner : m_runners) {
                    runner->terminate();
                }
                m_processed_chunks.terminate();
            }
            return;
        }

        // With the lock held, grab all the chunks we can accommodate in the
        // current batch from the chunk queue, but don't yet pass them to
        // the model input tensors.  We do this to minimise the time we need to
        // hold the mutex, which is highly contended, without having to repeatedly
        // lock/unlock, which is expensive enough in itself to slow down this thread
        // significantly.  This matters because slack time in this thread currently
        // gates Remora model GPU throughput on fast systems.
        size_t previous_chunk_count = batched_chunks.size();
        {
            nvtx3::scoped_range range{"push_chunks"};
            while (batched_chunks.size() != m_batch_size && !chunk_queue.empty()) {
                std::shared_ptr<RemoraChunk> chunk = chunk_queue.front();
                chunk_queue.pop_front();
                batched_chunks.push_back(chunk);
                last_chunk_reserve_time = std::chrono::system_clock::now();
            }
        }
        // Relinquish the chunk queue mutex, allowing other chunk queue
        // activity to progress.
        chunks_lock.unlock();
        m_chunk_queues_cv.notify_one();

        // Insert the chunks we just obtained into the model input tensors.
        for (size_t chunk_idx = previous_chunk_count; chunk_idx < batched_chunks.size();
             ++chunk_idx) {
            const auto& chunk = batched_chunks[chunk_idx];
            runner->accept_chunk(caller_id, chunk_idx, chunk->signal, chunk->encoded_kmers);
        }

        if (batched_chunks.size() == m_batch_size) {
            // Input tensor is full, let's get_scores.
            call_current_batch(worker_id, caller_id, batched_chunks);
        }
    }
}

void ModBaseCallerNode::call_current_batch(
        size_t worker_id,
        size_t caller_id,
        std::vector<std::shared_ptr<RemoraChunk>>& batched_chunks) {
    nvtx3::scoped_range loop{"call_current_batch"};

    dorado::stats::Timer timer;
    auto results = m_runners[worker_id]->call_chunks(caller_id, batched_chunks.size());
    m_call_chunks_ms += timer.GetElapsedMS();

    // Convert results to float32 with one call and address via a raw pointer,
    // to avoid huge libtorch indexing overhead.
    auto results_f32 = results.to(torch::kFloat32);
    assert(results_f32.is_contiguous());
    const auto* const results_f32_ptr = results_f32.data_ptr<float>();

    auto row_size = results.size(1);

    // Put results into chunk
    for (size_t i = 0; i < batched_chunks.size(); ++i) {
        auto& chunk = batched_chunks[i];
        chunk->scores.resize(row_size);
        std::memcpy(chunk->scores.data(), &results_f32_ptr[i * row_size], row_size * sizeof(float));
        m_processed_chunks.try_push(std::move(chunk));
    }

    batched_chunks.clear();
    ++m_num_batches_called;
}

void ModBaseCallerNode::output_worker_thread() {
    // The m_processed_chunks lock is sufficiently contended that it's worth taking all
    // chunks available once we obtain it.
    std::vector<std::shared_ptr<RemoraChunk>> processed_chunks;
    auto grab_chunk = [&processed_chunks](std::shared_ptr<RemoraChunk>& chunk) {
        processed_chunks.push_back(std::move(chunk));
    };
    while (m_processed_chunks.process_and_pop_all(grab_chunk)) {
        nvtx3::scoped_range range{"modbase_output_worker_thread"};

        for (const auto& chunk : processed_chunks) {
            auto source_read = chunk->source_read.lock();
            int64_t result_pos = chunk->context_hit;
            int64_t offset =
                    m_base_prob_offsets[RemoraUtils::BASE_IDS[source_read->seq[result_pos]]];
            for (size_t i = 0; i < chunk->scores.size(); ++i) {
                source_read->base_mod_probs[m_num_states * result_pos + offset + i] =
                        static_cast<uint8_t>(std::min(std::floor(chunk->scores[i] * 256), 255.0f));
            }
            ++source_read->num_modbase_chunks_called;
        }
        processed_chunks.clear();

        // Now move any completed reads to the output queue
        std::vector<std::shared_ptr<Read>> completed_reads;
        std::unique_lock<std::mutex> working_reads_lock(m_working_reads_mutex);
        for (auto read_iter = m_working_reads.begin(); read_iter != m_working_reads.end();) {
            if ((*read_iter)->num_modbase_chunks_called.load() ==
                (*read_iter)->num_modbase_chunks) {
                completed_reads.push_back(*read_iter);
                read_iter = m_working_reads.erase(read_iter);
            } else {
                ++read_iter;
            }
        }
        working_reads_lock.unlock();
        for (auto& read : completed_reads) {
            send_message_to_sink(read);
            ++m_num_mod_base_reads_pushed;
        }
    }
}

std::unordered_map<std::string, double> ModBaseCallerNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    for (const auto& runner : m_runners) {
        const auto runner_stats = stats::from_obj(*runner);
        stats.insert(runner_stats.begin(), runner_stats.end());
    }
    stats["batches_called"] = m_num_batches_called;
    stats["partial_batches_called"] = m_num_partial_batches_called;
    stats["input_chunks_sleeps"] = m_num_input_chunks_sleeps;
    stats["call_chunks_ms"] = m_call_chunks_ms;
    stats["context_hits"] = m_num_context_hits;
    stats["mod_base_reads_pushed"] = m_num_mod_base_reads_pushed;
    stats["non_mod_base_reads_pushed"] = m_num_non_mod_base_reads_pushed;
    stats["chunk_generation_ms"] = m_chunk_generation_ms;
    return stats;
}

}  // namespace dorado
