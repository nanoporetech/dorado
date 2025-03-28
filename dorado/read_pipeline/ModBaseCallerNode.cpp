#include "ModBaseCallerNode.h"

#include "config/ModBaseModelConfig.h"
#include "modbase/ModBaseContext.h"
#include "modbase/ModBaseRunner.h"
#include "modbase/ModbaseEncoder.h"
#include "torch_utils/tensor_utils.h"
#include "utils/math_utils.h"
#include "utils/sequence_utils.h"
#include "utils/stats.h"
#include "utils/thread_naming.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstring>
#include <iostream>

using namespace std::chrono_literals;

namespace dorado {

constexpr auto FORCE_TIMEOUT = 100ms;

struct ModBaseCallerNode::ModBaseChunk {
    ModBaseChunk(std::shared_ptr<WorkingRead> read,
                 at::Tensor input_signal,
                 std::vector<int8_t> kmer_data,
                 size_t position,
                 bool template_direction)
            : working_read(std::move(read)),
              signal(std::move(input_signal)),
              encoded_kmers(std::move(kmer_data)),
              context_hit(position),
              is_template_direction(template_direction) {}

    std::shared_ptr<WorkingRead> working_read;
    at::Tensor signal;
    std::vector<int8_t> encoded_kmers;
    size_t context_hit;
    std::vector<float> scores;
    bool is_template_direction;
};

struct ModBaseCallerNode::WorkingRead {
    Message read;  // The read itself.
    size_t num_modbase_chunks;
    std::atomic_size_t
            num_modbase_chunks_called;  // Number of modbase chunks which have been scored
};

ModBaseCallerNode::ModBaseCallerNode(std::vector<modbase::RunnerPtr> model_runners,
                                     size_t modbase_threads,
                                     size_t block_stride,
                                     size_t max_reads)
        : MessageSink(max_reads, static_cast<int>(modbase_threads)),
          m_runners(std::move(model_runners)),
          m_block_stride(block_stride),
          m_batch_size(m_runners[0]->batch_size()),
          // TODO -- more principled calculation of output queue size
          m_processed_chunks(10 * max_reads) {
    init_modbase_info();
    for (size_t i = 0; i < m_runners[0]->num_models(); i++) {
        m_chunk_queues.emplace_back(
                std::make_unique<utils::AsyncQueue<std::unique_ptr<ModBaseChunk>>>(m_batch_size *
                                                                                   5));
    }
}

ModBaseCallerNode::~ModBaseCallerNode() { terminate_impl(); }

void ModBaseCallerNode::start_threads() {
    m_output_worker = std::thread([this] { output_worker_thread(); });

    for (size_t worker_id = 0; worker_id < m_runners.size(); ++worker_id) {
        for (size_t model_id = 0; model_id < m_runners[worker_id]->num_models(); ++model_id) {
            m_runner_workers.emplace_back([=] { modbasecall_worker_thread(worker_id, model_id); });
            ++m_num_active_runner_workers;
        }
    }
    start_input_processing([this] { input_thread_fn(); }, "modbase_node");
}

void ModBaseCallerNode::terminate_impl() {
    // Signal termination in the input queue, and wait for input threads to join.
    stop_input_processing();
    // Signal termination in the chunk queues.
    for (auto& chunk_queue : m_chunk_queues) {
        chunk_queue->terminate();
    }
    // Wait for runner workers to join, now that they have been asked to via chunk queue
    // termination.
    for (auto& t : m_runner_workers) {
        t.join();
    }
    m_runner_workers.clear();
    if (m_output_worker.joinable()) {
        m_output_worker.join();
    }

    // There should be no reads left in the node after it's terminated.
    if (!m_working_reads.empty()) {
        throw std::logic_error("Reads have been left in ModBaseCallerNode");
    }
}

void ModBaseCallerNode::restart() {
    m_processed_chunks.restart();
    for (auto& runner : m_runners) {
        runner->restart();
    }
    for (auto& chunk_queue : m_chunk_queues) {
        chunk_queue->restart();
    }
    start_threads();
}

void ModBaseCallerNode::init_modbase_info() {
    std::vector<std::reference_wrapper<const config::ModBaseModelConfig>> base_mod_params;
    auto& runner = m_runners[0];
    modbase::ModBaseContext context_handler;
    for (size_t caller_id = 0; caller_id < runner->num_models(); ++caller_id) {
        const auto& params = runner->model_params(caller_id).mods;
        if (!params.motif.empty()) {
            context_handler.set_context(params.motif, size_t(params.motif_offset));
        }
        base_mod_params.emplace_back(runner->model_params(caller_id));
        m_num_states += params.count;
    }

    auto mod_info = config::get_modbase_info(base_mod_params);
    m_mod_base_info = std::make_shared<ModBaseInfo>(
            std::move(mod_info.alphabet), std::move(mod_info.long_names), context_handler.encode());
    m_base_prob_offsets = mod_info.base_probs_offsets();
}

void ModBaseCallerNode::duplex_mod_call(Message&& message) {
    auto read = std::get<DuplexReadPtr>(std::move(message));
    stats::Timer timer;

    {
        nvtx3::scoped_range range{"base_mod_probs_init"};
        // initialize base_mod_probs _before_ we start handing out chunks
        read->read_common.base_mod_probs.resize(read->read_common.seq.size() * m_num_states, 0);
        for (size_t i = 0; i < read->read_common.seq.size(); ++i) {
            // Initialize for what corresponds to 100% canonical base for each position.
            int base_id = utils::BaseInfo::BASE_IDS[read->read_common.seq[i]];
            if (base_id < 0) {
                throw std::runtime_error("Invalid character in sequence.");
            }
            read->read_common.base_mod_probs[i * m_num_states + m_base_prob_offsets[base_id]] = 1;
        }
    }

    read->read_common.mod_base_info = m_mod_base_info;

    try {
        auto working_read = std::make_shared<WorkingRead>();
        working_read->num_modbase_chunks = 0;
        working_read->num_modbase_chunks_called = 0;

        // all runners have the same set of callers, so we only need to use the first one
        auto& runner = m_runners[0];
        std::vector<std::vector<std::unique_ptr<ModBaseChunk>>> chunks_to_enqueue_by_caller(
                runner->num_models());

        std::vector<unsigned long> all_context_hits;

        for (const bool is_template_direction : {true, false}) {
            auto simplex_signal =
                    is_template_direction
                            ? read->stereo_feature_inputs.template_signal
                            : at::flip(read->stereo_feature_inputs.complement_signal, 0);

            // const-ref extends lifetime of temporary
            const auto& simplex_moves = is_template_direction
                                                ? read->stereo_feature_inputs.template_moves
                                                : read->stereo_feature_inputs.complement_moves;

            // const-ref extends lifetime of temporary
            const auto& simplex_seq =
                    is_template_direction
                            ? read->stereo_feature_inputs.template_seq
                            : utils::reverse_complement(read->stereo_feature_inputs.complement_seq);

            // const-ref extends lifetime of temporary
            const auto& duplex_seq = is_template_direction
                                             ? read->read_common.seq
                                             : utils::reverse_complement(read->read_common.seq);

            auto [moves_offset, target_start, new_move_table] =
                    utils::realign_moves(simplex_seq, duplex_seq, simplex_moves);

            // If the alignment has failed, the rest of this duplex mod call cannot be completed in this direction
            if (moves_offset == -1 && target_start == -1 && new_move_table.empty()) {
                continue;
            }

            auto signal_len = new_move_table.size() * m_block_stride;
            auto num_moves = std::accumulate(new_move_table.begin(), new_move_table.end(), 0);
            auto new_seq = duplex_seq.substr(target_start, num_moves);
            std::vector<int> sequence_ints = utils::sequence_to_ints(new_seq);

            // no reverse_signal in duplex, so we can do this once for all callers
            std::vector<uint64_t> seq_to_sig_map =
                    utils::moves_to_map(new_move_table, m_block_stride, signal_len, num_moves + 1);

            for (size_t caller_id = 0; caller_id < runner->num_models(); ++caller_id) {
                nvtx3::scoped_range range{"generate_chunks"};
                auto& chunks_to_enqueue = chunks_to_enqueue_by_caller.at(caller_id);
                auto& params = runner->model_params(caller_id);
                auto signal = simplex_signal.slice(0, moves_offset * m_block_stride,
                                                   moves_offset * m_block_stride + signal_len);

                // scale signal based on model parameters
                auto scaled_signal =
                        runner->scale_signal(caller_id, signal, sequence_ints, seq_to_sig_map);

                // One-hot encodes the kmer at each signal step for input into the network
                modbase::ModBaseEncoder encoder(
                        m_block_stride, params.context.samples, params.context.bases_before,
                        params.context.bases_after, params.context.base_start_justify);
                encoder.init(sequence_ints, seq_to_sig_map);

                auto context_hits = runner->get_motif_hits(caller_id, new_seq);
                m_num_context_hits += static_cast<int64_t>(context_hits.size());
                chunks_to_enqueue.reserve(context_hits.size());

                for (auto context_hit : context_hits) {
                    nvtx3::scoped_range range_create_chunk{"create_chunk"};
                    auto slice = encoder.get_context(context_hit);
                    // signal
                    auto input_signal = scaled_signal.index({at::indexing::Slice(
                            slice.first_sample, slice.first_sample + slice.num_existing_samples)});
                    if (slice.lead_samples_needed != 0 || slice.tail_samples_needed != 0) {
                        input_signal = at::constant_pad_nd(input_signal,
                                                           {(int64_t)slice.lead_samples_needed,
                                                            (int64_t)slice.tail_samples_needed});
                    }

                    // Update the context hit into the duplex reference context
                    unsigned long context_hit_in_duplex_space;
                    if (is_template_direction) {
                        context_hit_in_duplex_space =
                                static_cast<unsigned long>(context_hit + target_start);
                    } else {
                        context_hit_in_duplex_space = static_cast<unsigned long>(
                                read->read_common.seq.size() - (context_hit + target_start + 1));
                    }

                    chunks_to_enqueue.push_back(std::make_unique<ModBaseChunk>(
                            working_read, input_signal, std::move(slice.data),
                            context_hit_in_duplex_space, is_template_direction));

                    all_context_hits.push_back(context_hit_in_duplex_space);
                    ++working_read->num_modbase_chunks;
                }
            }
        }

        m_chunk_generation_ms += timer.GetElapsedMS();

        if (working_read->num_modbase_chunks != 0) {
            // Hand over our ownership to the working read
            working_read->read = std::move(read);

            // Put the read in the working list
            {
                std::lock_guard<std::mutex> working_reads_lock(m_working_reads_mutex);
                m_working_reads.insert(std::move(working_read));
                ++m_working_reads_size;
            }

            // push the chunks to the chunk queues
            // needs to be done after working_read->read is set as chunks could be processed
            // before we set that value otherwise
            for (size_t caller_id = 0; caller_id < runner->num_models(); ++caller_id) {
                auto& chunk_queue = m_chunk_queues.at(caller_id);
                auto& chunks_to_enqueue = chunks_to_enqueue_by_caller.at(caller_id);
                for (auto& chunk : chunks_to_enqueue) {
                    chunk_queue->try_push(std::move(chunk));
                }
            }
        } else {
            // No modbases to call, pass directly to next node
            send_message_to_sink(std::move(read));
            ++m_num_non_mod_base_reads_pushed;
        }
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
    }
}

void ModBaseCallerNode::simplex_mod_call(Message&& message) {
    auto read = std::get<SimplexReadPtr>(std::move(message));
    stats::Timer timer;
    {
        nvtx3::scoped_range range{"base_mod_probs_init"};
        // initialize base_mod_probs _before_ we start handing out chunks
        read->read_common.base_mod_probs.resize(read->read_common.seq.size() * m_num_states, 0);
        for (size_t i = 0; i < read->read_common.seq.size(); ++i) {
            // Initialize for what corresponds to 100% canonical base for each position.
            int base_id = utils::BaseInfo::BASE_IDS[read->read_common.seq[i]];
            if (base_id < 0) {
                throw std::runtime_error("Invalid character in sequence.");
            }
            read->read_common.base_mod_probs[i * m_num_states + m_base_prob_offsets[base_id]] = 1;
        }
    }
    read->read_common.mod_base_info = m_mod_base_info;

    auto working_read = std::make_shared<WorkingRead>();
    working_read->num_modbase_chunks = 0;
    working_read->num_modbase_chunks_called = 0;

    std::vector<int> sequence_ints = utils::sequence_to_ints(read->read_common.seq);

    // all runners have the same set of callers, so we only need to use the first one
    auto& runner = m_runners[0];
    std::vector<std::vector<std::unique_ptr<ModBaseChunk>>> chunks_to_enqueue_by_caller(
            runner->num_models());
    for (size_t caller_id = 0; caller_id < runner->num_models(); ++caller_id) {
        nvtx3::scoped_range range{"generate_chunks"};

        auto signal_len = read->read_common.get_raw_data_samples();
        std::vector<uint64_t> seq_to_sig_map =
                utils::moves_to_map(read->read_common.moves, m_block_stride, signal_len,
                                    read->read_common.seq.size() + 1);

        auto& chunks_to_enqueue = chunks_to_enqueue_by_caller.at(caller_id);
        auto& params = runner->model_params(caller_id);
        auto signal = read->read_common.raw_data;
        if (params.context.reverse) {
            signal = at::flip(signal, 0);
            std::reverse(std::begin(seq_to_sig_map), std::end(seq_to_sig_map));
            std::transform(std::begin(seq_to_sig_map), std::end(seq_to_sig_map),
                           std::begin(seq_to_sig_map),
                           [signal_len](auto signal_pos) { return signal_len - signal_pos; });
        }

        // scale signal based on model parameters
        auto scaled_signal = runner->scale_signal(caller_id, signal, sequence_ints, seq_to_sig_map);

        // One-hot encodes the kmer at each signal step for input into the network
        modbase::ModBaseEncoder encoder(m_block_stride, params.context.samples,
                                        params.context.bases_before, params.context.bases_after,
                                        params.context.base_start_justify);
        encoder.init(sequence_ints, seq_to_sig_map);

        auto context_hits = runner->get_motif_hits(caller_id, read->read_common.seq);
        m_num_context_hits += static_cast<int64_t>(context_hits.size());
        chunks_to_enqueue.reserve(context_hits.size());
        for (auto context_hit : context_hits) {
            nvtx3::scoped_range nvtxrange{"create_chunk"};
            auto slice = encoder.get_context(context_hit);
            // signal
            auto input_signal = scaled_signal.index({at::indexing::Slice(
                    slice.first_sample, slice.first_sample + slice.num_existing_samples)});
            if (slice.lead_samples_needed != 0 || slice.tail_samples_needed != 0) {
                input_signal = at::constant_pad_nd(
                        input_signal,
                        {(int64_t)slice.lead_samples_needed, (int64_t)slice.tail_samples_needed});
            }
            chunks_to_enqueue.push_back(std::make_unique<ModBaseChunk>(
                    working_read, input_signal, std::move(slice.data), context_hit, true));

            ++working_read->num_modbase_chunks;
        }
    }
    m_chunk_generation_ms += timer.GetElapsedMS();

    if (working_read->num_modbase_chunks != 0) {
        // Hand over our ownership to the working read
        working_read->read = std::move(read);

        // Put the read in the working list
        {
            std::lock_guard<std::mutex> working_reads_lock(m_working_reads_mutex);
            m_working_reads.insert(std::move(working_read));
            ++m_working_reads_size;
        }

        // push the chunks to the chunk queues
        // needs to be done after working_read->read is set as chunks could be processed
        // before we set that value otherwise
        for (size_t caller_id = 0; caller_id < runner->num_models(); ++caller_id) {
            auto& chunk_queue = m_chunk_queues.at(caller_id);
            auto& chunks_to_enqueue = chunks_to_enqueue_by_caller.at(caller_id);
            for (auto& chunk : chunks_to_enqueue) {
                chunk_queue->try_push(std::move(chunk));
            }
        }
    } else {
        // No modbases to call, pass directly to next node
        send_message_to_sink(std::move(read));
        ++m_num_non_mod_base_reads_pushed;
    }
}

void ModBaseCallerNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!is_read_message(message)) {
            send_message_to_sink(std::move(message));
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            simplex_mod_call(std::move(message));
        } else if (std::holds_alternative<DuplexReadPtr>(message)) {
            duplex_mod_call(std::move(message));
        }
    }
}

void ModBaseCallerNode::modbasecall_worker_thread(size_t worker_id, size_t caller_id) {
    utils::set_thread_name("modbase_worker");
    at::InferenceMode inference_mode_guard;

    auto& runner = m_runners[worker_id];
    auto& chunk_queue = m_chunk_queues[caller_id];

    std::vector<std::unique_ptr<ModBaseChunk>> batched_chunks;
    auto last_chunk_reserve_time = std::chrono::system_clock::now();

    size_t previous_chunk_count = 0;
    while (true) {
        nvtx3::scoped_range range{"modbasecall_worker_thread"};
        // Repeatedly attempt to complete the current batch with one acquisition of the
        // chunk queue mutex.
        auto grab_chunk = [&batched_chunks](std::unique_ptr<ModBaseChunk> chunk) {
            batched_chunks.push_back(std::move(chunk));
        };
        const auto status = chunk_queue->process_and_pop_n_with_timeout(
                grab_chunk, m_batch_size - batched_chunks.size(),
                last_chunk_reserve_time + FORCE_TIMEOUT);
        if (status == utils::AsyncQueueStatus::Terminate) {
            break;
        }

        // Reset timeout.
        last_chunk_reserve_time = std::chrono::system_clock::now();

        // We have just grabbed a number of chunks (0 in the case of timeout) from
        // the chunk queue and added them to batched_chunks.  Insert those chunks
        // into the model input tensors.
        for (size_t chunk_idx = previous_chunk_count; chunk_idx < batched_chunks.size();
             ++chunk_idx) {
            assert(chunk_idx < m_batch_size);
            const auto& chunk = batched_chunks[chunk_idx];
            runner->accept_chunk(int(caller_id), int(chunk_idx), chunk->signal,
                                 chunk->encoded_kmers);
        }

        // If we have a complete batch, or we have a partial batch and timed out,
        // then call what we have.
        if (batched_chunks.size() == m_batch_size ||
            (status == utils::AsyncQueueStatus::Timeout && !batched_chunks.empty())) {
            // Input tensor is full, let's get scores.
            call_current_batch(worker_id, caller_id, batched_chunks);
        }

        previous_chunk_count = batched_chunks.size();
    }

    // Basecall any remaining chunks.
    if (!batched_chunks.empty()) {
        call_current_batch(worker_id, caller_id, batched_chunks);
    }

    // Reduce the count of active model callers.  If this was the last active
    // model caller also send termination signal to sink
    int num_remaining_callers = --m_num_active_runner_workers;
    if (num_remaining_callers == 0) {
        m_processed_chunks.terminate();
    }
}

void ModBaseCallerNode::call_current_batch(
        size_t worker_id,
        size_t caller_id,
        std::vector<std::unique_ptr<ModBaseChunk>>& batched_chunks) {
    nvtx3::scoped_range loop{"call_current_batch"};

    dorado::stats::Timer timer;
    auto results = m_runners[worker_id]->call_chunks(int(caller_id), int(batched_chunks.size()));
    m_call_chunks_ms += timer.GetElapsedMS();

    // Convert results to float32 with one call and address via a raw pointer,
    // to avoid huge libtorch indexing overhead.
    auto results_f32 = results.to(at::ScalarType::Float);
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

    if (batched_chunks.size() == m_batch_size) {
        ++m_num_batches_called;
    } else {
        ++m_num_partial_batches_called;
    }

    batched_chunks.clear();
}

void ModBaseCallerNode::output_worker_thread() {
    utils::set_thread_name("modbase_out");
    at::InferenceMode inference_mode_guard;

    // The m_processed_chunks lock is sufficiently contended that it's worth taking all
    // chunks available once we obtain it.
    std::vector<std::unique_ptr<ModBaseChunk>> processed_chunks;
    auto grab_chunk = [&processed_chunks](std::unique_ptr<ModBaseChunk> chunk) {
        processed_chunks.push_back(std::move(chunk));
    };
    while (m_processed_chunks.process_and_pop_n(grab_chunk, m_processed_chunks.capacity()) ==
           utils::AsyncQueueStatus::Success) {
        nvtx3::scoped_range range{"modbase_output_worker_thread"};

        std::vector<std::shared_ptr<WorkingRead>> completed_reads;

        for (const auto& chunk : processed_chunks) {
            auto working_read = chunk->working_read;
            auto& source_read = working_read->read;
            auto& source_read_common = get_read_common_data(source_read);

            int64_t result_pos = chunk->context_hit;

            int64_t offset;
            const auto& baseIds = utils::BaseInfo::BASE_IDS;
            const auto& seq = source_read_common.seq[result_pos];

            offset = chunk->is_template_direction
                             ? m_base_prob_offsets[baseIds[seq]]
                             : m_base_prob_offsets[baseIds[dorado::utils::complement_table[seq]]];

            auto num_chunk_scores = chunk->scores.size();
            for (size_t i = 0; i < num_chunk_scores; ++i) {
                source_read_common.base_mod_probs[m_num_states * result_pos + offset + i] =
                        static_cast<uint8_t>(std::min(std::floor(chunk->scores[i] * 256), 255.0f));
            }
            // If all chunks for the read associated with this chunk have now been called,
            // add it to the completed_reads vector for subsequent sending on to the sink.
            auto num_chunks_called = ++working_read->num_modbase_chunks_called;
            if (num_chunks_called == working_read->num_modbase_chunks) {
                completed_reads.push_back(std::move(working_read));
            }
        }
        processed_chunks.clear();

        // Remove any completed reads from the working reads set while holding its mutex.
        if (!completed_reads.empty()) {
            std::lock_guard<std::mutex> working_reads_lock(m_working_reads_mutex);
            for (auto& completed_read : completed_reads) {
                auto read_iter = m_working_reads.find(completed_read);
                if (read_iter != m_working_reads.end()) {
                    m_working_reads.erase(read_iter);
                } else {
                    auto read_id = get_read_common_data(completed_read->read).read_id;
                    throw std::runtime_error("Expected to find read id " + read_id +
                                             " in working reads set but it doesn't exist.");
                }
            }
            m_working_reads_size -= completed_reads.size();
        }

        // Send completed reads on to the sink.
        for (auto& completed_read : completed_reads) {
            send_message_to_sink(std::move(completed_read->read));
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
    stats["batches_called"] = double(m_num_batches_called);
    stats["partial_batches_called"] = double(m_num_partial_batches_called);
    stats["input_chunks_sleeps"] = double(m_num_input_chunks_sleeps);
    stats["call_chunks_ms"] = double(m_call_chunks_ms);
    stats["context_hits"] = double(m_num_context_hits);
    stats["mod_base_reads_pushed"] = double(m_num_mod_base_reads_pushed);
    stats["non_mod_base_reads_pushed"] = double(m_num_non_mod_base_reads_pushed);
    stats["chunk_generation_ms"] = double(m_chunk_generation_ms);
    stats["working_reads_items"] = double(m_working_reads_size);
    return stats;
}

}  // namespace dorado
