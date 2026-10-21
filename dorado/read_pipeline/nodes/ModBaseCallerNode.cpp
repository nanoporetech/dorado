#include "read_pipeline/nodes/ModBaseCallerNode.h"

#include "config/ModBaseModelConfig.h"
#include "modbase/ModBaseContext.h"
#include "modbase/ModBaseEncoder.h"
#include "modbase/ModBaseRunner.h"
#include "read_pipeline/base/messages/DuplexRead.h"
#include "read_pipeline/base/messages/SimplexRead.h"
#include "utils/log_utils.h"
#include "utils/math_utils.h"
#include "utils/sequence_utils.h"
#include "utils/stats.h"
#include "utils/thread_utils.h"

#include <ATen/TensorIndexing.h>
#include <nvtx3/nvtx3.hpp>

#include <chrono>
#include <cstring>
#include <numeric>
#include <utility>

using namespace std::chrono_literals;

namespace dorado {

namespace {

class SimplexReadPolicy {
public:
    explicit SimplexReadPolicy(ReadCommon& read) : m_read(read) {
        m_read.base_mod_simplex_motif_hits.resize(m_read.seq.size(), false);
    }

    bool next_strand() { return !std::exchange(m_processed, true); }

    const at::Tensor& signal() const { return m_read.raw_data; }
    const std::string& sequence() const { return m_read.seq; }
    const std::vector<uint8_t>& moves() const { return m_read.moves; }
    bool is_template_direction() const { return true; }

    void adapt_model_input(const config::modbase::ModBaseModelConfigData& params,
                           at::Tensor& signal,
                           std::vector<uint64_t>& seq_to_sig_map,
                           size_t signal_len) const {
        if (params.context.reverse) {
            signal = at::flip(signal, 0);
            utils::reverse_seq_to_sig_map(seq_to_sig_map, signal_len);
        }
    }

    size_t map_hit(size_t context_hit) const { return context_hit; }

    void observe_hit(size_t context_hit, size_t) {
        m_read.base_mod_simplex_motif_hits.at(context_hit) = true;
    }

private:
    ReadCommon& m_read;
    bool m_processed{false};
};

class DuplexReadPolicy {
public:
    DuplexReadPolicy(DuplexRead& read, size_t block_stride)
            : m_read(read), m_block_stride(block_stride) {}

    bool next_strand() {
        while (++m_strand_index < 2) {
            m_is_template_direction = m_strand_index == 0;
            const auto& stereo_inputs = m_read.stereo_feature_inputs;

            auto simplex_signal = m_is_template_direction
                                          ? stereo_inputs.template_signal
                                          : at::flip(stereo_inputs.complement_signal, 0);
            const auto& simplex_moves = m_is_template_direction ? stereo_inputs.template_moves
                                                                : stereo_inputs.complement_moves;
            const auto simplex_sequence =
                    m_is_template_direction
                            ? stereo_inputs.template_seq
                            : utils::reverse_complement(stereo_inputs.complement_seq);
            const auto duplex_sequence =
                    m_is_template_direction ? m_read.read_common.seq
                                            : utils::reverse_complement(m_read.read_common.seq);

            auto [moves_offset, target_start, realigned_moves] =
                    utils::realign_moves(simplex_sequence, duplex_sequence, simplex_moves);
            if (moves_offset == -1 && target_start == -1 && realigned_moves.empty()) {
                continue;
            }

            const size_t signal_len = realigned_moves.size() * m_block_stride;
            const auto num_moves =
                    std::accumulate(realigned_moves.begin(), realigned_moves.end(), 0);
            m_sequence = duplex_sequence.substr(target_start, num_moves);
            m_signal = simplex_signal.slice(0, moves_offset * m_block_stride,
                                            moves_offset * m_block_stride + signal_len);
            m_moves = std::move(realigned_moves);
            m_target_start = target_start;
            return true;
        }
        return false;
    }

    const at::Tensor& signal() const { return m_signal; }
    const std::string& sequence() const { return m_sequence; }
    const std::vector<uint8_t>& moves() const { return m_moves; }
    bool is_template_direction() const { return m_is_template_direction; }

    void adapt_model_input(const config::modbase::ModBaseModelConfigData&,
                           at::Tensor&,
                           std::vector<uint64_t>&,
                           size_t) const {}

    size_t map_hit(size_t context_hit) const {
        if (m_is_template_direction) {
            return context_hit + m_target_start;
        }
        return m_read.read_common.seq.size() - (context_hit + m_target_start + 1);
    }

    void observe_hit(size_t, size_t) const {}

private:
    DuplexRead& m_read;
    const size_t m_block_stride;
    int m_strand_index{-1};
    bool m_is_template_direction{true};
    int64_t m_target_start{0};
    at::Tensor m_signal;
    std::string m_sequence;
    std::vector<uint8_t> m_moves;
};

}  // namespace

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

    m_processed_chunks.set_name("processed_chunks");
    for (size_t i = 0; i < m_runners[0]->num_models(); i++) {
        auto& queue = m_chunk_queues.emplace_back(
                std::make_unique<utils::AsyncQueue<std::unique_ptr<ModBaseChunk>>>(m_batch_size *
                                                                                   5));
        queue->set_name("chunk_queue_" + std::to_string(i));
    }
}

ModBaseCallerNode::~ModBaseCallerNode() { terminate_impl(utils::AsyncQueueTerminateFast::Yes); }

void ModBaseCallerNode::start_threads() {
    m_output_worker = std::thread([this] { output_worker_thread(); });

    for (size_t worker_id = 0; worker_id < m_runners.size(); ++worker_id) {
        for (size_t model_id = 0; model_id < m_runners[worker_id]->num_models(); ++model_id) {
            m_runner_workers.emplace_back([this, worker_id, model_id] {
                modbasecall_worker_thread(worker_id, model_id);
            });
        }
    }
    start_input_processing([this] { input_thread_fn(); }, "modbase_node");
}

void ModBaseCallerNode::terminate_impl(utils::AsyncQueueTerminateFast fast) {
    // Signal termination in the input queue, and wait for input threads to join.
    stop_input_processing(fast);

    // Signal termination in the chunk queues.
    for (auto& chunk_queue : m_chunk_queues) {
        chunk_queue->terminate(fast);
    }
    // Wait for runner workers to join, now that they have been asked to via chunk queue
    // termination.
    for (auto& t : m_runner_workers) {
        t.join();
    }
    m_runner_workers.clear();

    // Signal the output thread to terminate.
    m_processed_chunks.terminate(fast);
    if (m_output_worker.joinable()) {
        m_output_worker.join();
    }

    // There should be no reads left in the node after it's terminated.
    // Unless we're terminating fast, in which case we can safely drop them.
    if (fast == utils::AsyncQueueTerminateFast::Yes) {
        m_working_reads.clear();
    } else if (!m_working_reads.empty()) {
        throw std::logic_error("Reads have been left in ModBaseCallerNode");
    }
}

std::string ModBaseCallerNode::get_name() const { return "ModBaseCallerNode"; }

void ModBaseCallerNode::terminate(const TerminateOptions& terminate_options) {
    terminate_impl(terminate_options.fast);
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
        const auto& params = config::get_model_config_data(runner->model_params(caller_id)).mods;
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

void ModBaseCallerNode::initialise_base_mod_probs(ReadCommon& read) const {
    nvtx3::scoped_range range{"base_mod_probs_init"};

    // Initialise for what corresponds to 100% canonical base for each position.
    read.base_mod_probs.resize(read.seq.size() * m_num_states, 0);
    for (size_t i = 0; i < read.seq.size(); ++i) {
        const auto base_id = utils::BaseInfo::BASE_IDS.at(static_cast<std::uint8_t>(read.seq[i]));
        if (base_id < 0) {
            throw std::runtime_error("Invalid character in sequence.");
        }
        read.base_mod_probs[i * m_num_states + m_base_prob_offsets[base_id]] = 1;
    }
    read.mod_base_info = m_mod_base_info;
}

void ModBaseCallerNode::submit_or_forward(
        Message message,
        std::shared_ptr<WorkingRead> working_read,
        std::vector<std::vector<std::unique_ptr<ModBaseChunk>>>& chunks_by_caller) {
    if (working_read->num_modbase_chunks == 0) {
        send_message_to_sink(std::move(message));
        ++m_num_non_mod_base_reads_pushed;
        return;
    }

    // Hand over our ownership to the working read.
    working_read->read = std::move(message);

    // Put the read in the working list.
    {
        std::lock_guard<std::mutex> working_reads_lock(m_working_reads_mutex);
        m_working_reads.insert(std::move(working_read));
        ++m_working_reads_size;
    }

    // Push the chunks to the chunk queues. This needs to be done after working_read->read is set,
    // as chunks could be processed before that value is set otherwise.
    for (size_t caller_id = 0; caller_id < chunks_by_caller.size(); ++caller_id) {
        auto& chunk_queue = m_chunk_queues.at(caller_id);
        for (auto& chunk : chunks_by_caller.at(caller_id)) {
            chunk_queue->try_push(std::move(chunk));
        }
    }
}

template <typename ReadPolicy>
void ModBaseCallerNode::generate_chunks_for_strand(
        ReadPolicy& policy,
        const std::shared_ptr<WorkingRead>& working_read,
        modbase::ModBaseRunner& runner,
        std::vector<std::vector<std::unique_ptr<ModBaseChunk>>>& chunks_by_caller) {
    const std::vector<int> sequence_ints = utils::sequence_to_ints(policy.sequence());
    const size_t signal_len = policy.signal().size(0);
    const std::vector<uint64_t> base_seq_to_sig_map = utils::moves_to_map(
            policy.moves(), m_block_stride, signal_len, policy.sequence().size() + 1);

    for (size_t caller_id = 0; caller_id < runner.num_models(); ++caller_id) {
        nvtx3::scoped_range range{"generate_chunks"};
        auto& chunks_to_enqueue = chunks_by_caller.at(caller_id);
        const auto& params = config::get_context_model_config(runner.model_params(caller_id));
        auto signal = policy.signal();
        auto seq_to_sig_map = base_seq_to_sig_map;
        policy.adapt_model_input(params, signal, seq_to_sig_map, signal_len);

        auto scaled_signal = runner.scale_signal(caller_id, signal, sequence_ints, seq_to_sig_map);

        modbase::ModBaseEncoder encoder(m_block_stride, params.context.samples,
                                        params.context.bases_before, params.context.bases_after,
                                        params.context.base_start_justify);
        encoder.init(sequence_ints, seq_to_sig_map);

        auto context_hits = runner.get_motif_hits(caller_id, policy.sequence());
        m_num_context_hits += static_cast<int64_t>(context_hits.size());
        chunks_to_enqueue.reserve(chunks_to_enqueue.size() + context_hits.size());

        for (auto context_hit : context_hits) {
            nvtx3::scoped_range range_create_chunk{"create_chunk"};
            auto slice = encoder.get_context(context_hit);
            auto input_signal = scaled_signal.index({at::indexing::Slice(
                    slice.first_sample, slice.first_sample + slice.num_existing_samples)});
            if (slice.lead_samples_needed != 0 || slice.tail_samples_needed != 0) {
                input_signal = at::constant_pad_nd(
                        input_signal,
                        {(int64_t)slice.lead_samples_needed, (int64_t)slice.tail_samples_needed});
            }

            const size_t mapped_context_hit = policy.map_hit(context_hit);
            chunks_to_enqueue.push_back(std::make_unique<ModBaseChunk>(
                    working_read, input_signal, std::move(slice.data), mapped_context_hit,
                    policy.is_template_direction()));

            ++working_read->num_modbase_chunks;
            policy.observe_hit(context_hit, mapped_context_hit);
        }
    }
}

template <typename ReadPolicy>
void ModBaseCallerNode::process_read(Message message, ReadPolicy& policy, stats::Timer& timer) {
    auto working_read = std::make_shared<WorkingRead>();
    working_read->num_modbase_chunks = 0;
    working_read->num_modbase_chunks_called = 0;

    // All runners have the same set of callers, so only the first is needed for chunk generation.
    auto& runner = *m_runners.front();
    std::vector<std::vector<std::unique_ptr<ModBaseChunk>>> chunks_by_caller(runner.num_models());

    while (policy.next_strand()) {
        generate_chunks_for_strand(policy, working_read, runner, chunks_by_caller);
    }

    m_chunk_generation_ms += timer.GetElapsedMS();
    submit_or_forward(std::move(message), std::move(working_read), chunks_by_caller);
}

void ModBaseCallerNode::mod_call(Message message) {
    stats::Timer timer;
    auto& read_common = get_read_common_data(message);
    initialise_base_mod_probs(read_common);

    if (message.holds<SimplexReadPtr>()) {
        SimplexReadPolicy policy(read_common);
        process_read(std::move(message), policy, timer);
        return;
    }

    try {
        DuplexReadPolicy policy(*message.get<DuplexReadPtr>(), m_block_stride);
        process_read(std::move(message), policy, timer);
    } catch (const std::exception& e) {
        DORADO_LOG(error, e.what());
    }
}

void ModBaseCallerNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!is_read_message(message)) {
            send_message_to_sink(std::move(message));
        } else {
            mod_call(std::move(message));
        }
    }
}

void ModBaseCallerNode::modbasecall_worker_thread(size_t worker_id, size_t caller_id) {
    utils::set_thread_name("modbase_worker");
    at::InferenceMode inference_mode_guard;

    auto& runner = m_runners[worker_id];
    auto& chunk_queue = m_chunk_queues[caller_id];

    using Clock = std::remove_reference_t<decltype(chunk_queue)>::element_type::Clock;
    auto last_chunk_reserve_time = Clock::now();

    std::vector<std::unique_ptr<ModBaseChunk>> batched_chunks;
    batched_chunks.reserve(m_batch_size);
    while (true) {
        nvtx3::scoped_range range{"modbasecall_worker_thread"};

        // Repeatedly attempt to complete the current batch with one acquisition of the
        // chunk queue mutex.
        const std::size_t previous_chunk_count = batched_chunks.size();
        auto grab_chunk = [&batched_chunks](std::unique_ptr<ModBaseChunk> chunk) {
            batched_chunks.push_back(std::move(chunk));
        };
        const auto status = chunk_queue->process_and_pop_n_with_timeout(
                grab_chunk, m_batch_size - previous_chunk_count,
                last_chunk_reserve_time + FORCE_TIMEOUT);
        if (status == utils::AsyncQueueStatus::Terminate) {
            break;
        }

        // Reset timeout.
        last_chunk_reserve_time = Clock::now();

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
    }

    // Basecall any remaining chunks.
    if (!batched_chunks.empty()) {
        call_current_batch(worker_id, caller_id, batched_chunks);
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

    std::vector<std::shared_ptr<WorkingRead>> completed_reads;
    std::vector<std::unique_ptr<ModBaseChunk>> processed_chunks;
    processed_chunks.reserve(m_processed_chunks.capacity());

    // The m_processed_chunks lock is sufficiently contended that it's worth taking all
    // chunks available once we obtain it.
    auto grab_chunk = [&processed_chunks](std::unique_ptr<ModBaseChunk> chunk) {
        processed_chunks.push_back(std::move(chunk));
    };
    while (m_processed_chunks.process_and_pop_n(grab_chunk, m_processed_chunks.capacity()) ==
           utils::AsyncQueueStatus::Success) {
        nvtx3::scoped_range range{"modbase_output_worker_thread"};

        completed_reads.clear();
        for (const auto& chunk : processed_chunks) {
            auto working_read = std::move(chunk->working_read);
            auto& source_read = working_read->read;
            auto& source_read_common = get_read_common_data(source_read);

            int64_t result_pos = chunk->context_hit;

            const auto& baseIds = utils::BaseInfo::BASE_IDS;
            const std::uint8_t seq = source_read_common.seq[result_pos];
            const int64_t offset =
                    chunk->is_template_direction
                            ? m_base_prob_offsets[baseIds.at(seq)]
                            : m_base_prob_offsets[baseIds.at(dorado::utils::complement_table[seq])];

            const auto num_chunk_scores = chunk->scores.size();
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
    stats::NamedStats stats = MessageSink::sample_stats();
    for (const auto& runner : m_runners) {
        stats.merge(stats::from_obj(*runner));
    }
    for (const auto& chunk_queue : m_chunk_queues) {
        stats.merge(stats::from_obj(*chunk_queue));
    }
    stats.merge(stats::from_obj(m_processed_chunks));
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
