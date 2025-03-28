#include "ModBaseChunkCallerNode.h"

#include "config/ModBaseModelConfig.h"
#include "messages.h"
#include "modbase/ModBaseContext.h"
#include "modbase/encode_kmer.h"
#include "utils/dev_utils.h"
#include "utils/math_utils.h"
#include "utils/sequence_utils.h"
#include "utils/thread_naming.h"
#include "utils/types.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <torch/version.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using namespace std::chrono_literals;

}  // namespace

namespace dorado {

constexpr auto FORCE_TIMEOUT = 100ms;

struct ModBaseChunkCallerNode::ModBaseData {
    // Scaled/padded signal.
    at::Tensor signal;
    // Padded encoded kmers.
    std::vector<int8_t> encoded_kmers;
    // Sequence indices for hits for each base (i.e. one per model)
    PerBaseIntVec per_base_hits_seq;
    // Signal indices for hits for each base (i.e. one per model)
    PerBaseIntVec per_base_hits_sig;
    // The location in the target sequence where the realigned sequence starts - duplex only
    int64_t target_start{0};
};

struct ModBaseChunkCallerNode::WorkingRead {
    // Underlying read.
    Message read;
    // Template modbase data
    ModBaseData template_data;
    // Complement modbase data - **only valid for duplex**.
    // This data is the reverse complement of the stereo input.
    // This means that the signal and sequence are in the **template** strand direction
    ModBaseData complement_data;
    // True if this working read is from a duplex read
    bool is_duplex{false};
    // Number of modbase chunks to process.
    size_t num_modbase_chunks{0};
    // Number of modbase chunks that have been processed.
    std::atomic_size_t num_modbase_chunks_called{0};
};

ModBaseChunkCallerNode::ModBaseChunkCallerNode(std::vector<modbase::RunnerPtr> model_runners,
                                               size_t modbase_threads,
                                               size_t canonical_stride,
                                               size_t max_reads)
        : MessageSink(max_reads, static_cast<int>(modbase_threads)),
          m_runners(std::move(model_runners)),
          m_canonical_stride(canonical_stride),
          m_batch_size(m_runners.at(0)->batch_size()),
          m_kmer_len(m_runners.at(0)->model_params(0).context.kmer_len),
          m_is_rna_model(m_runners.at(0)->model_params(0).context.reverse),
          m_processed_chunks(m_runners.size() * 8 * m_batch_size),
          m_pad_end_align(utils::get_dev_opt<bool>("modbase_pad_end_align", 0)) {
    init_modbase_info();
    validate_runners();

    for (size_t i = 0; i < m_runners.at(0)->num_models(); ++i) {
        m_chunk_queues.push_back(std::make_unique<utils::AsyncQueue<std::unique_ptr<ModBaseChunk>>>(
                m_batch_size * 10));
    }
}

ModBaseChunkCallerNode::~ModBaseChunkCallerNode() { terminate_impl(); }

void ModBaseChunkCallerNode::start_threads() {
    m_output_workers.emplace_back([=] { output_thread_fn(); });

    for (size_t worker_id = 0; worker_id < m_runners.size(); ++worker_id) {
        for (size_t model_id = 0; model_id < m_runners[worker_id]->num_models(); ++model_id) {
            m_runner_workers.emplace_back([=] { chunk_caller_thread_fn(worker_id, model_id); });
            ++m_num_active_runner_workers;
        }
    }
    // This creates num_threads threads defined in the MessageSink(limit, num_threads)
    start_input_processing([this] { input_thread_fn(); }, "chunk_modbase_node");
}

void ModBaseChunkCallerNode::input_thread_fn() {
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

void ModBaseChunkCallerNode::terminate_impl() {
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

    for (auto& t : m_output_workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    m_output_workers.clear();

    // There should be no reads left in the node after it's terminated.
    if (!m_working_reads.empty()) {
        throw std::logic_error("Reads have been left in ModBaseChunkCallerNode");
    }
}

void ModBaseChunkCallerNode::restart() {
    m_processed_chunks.restart();
    for (auto& runner : m_runners) {
        runner->restart();
    }
    for (auto& chunk_queue : m_chunk_queues) {
        chunk_queue->restart();
    }
    start_threads();
}

void ModBaseChunkCallerNode::init_modbase_info() {
    std::vector<std::reference_wrapper<const config::ModBaseModelConfig>> base_mod_params;
    auto& runner = m_runners.at(0);
    modbase::ModBaseContext context_handler;
    for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
        const auto& params = runner->model_params(model_id).mods;
        if (!params.motif.empty()) {
            context_handler.set_context(params.motif, size_t(params.motif_offset));
        }
        base_mod_params.push_back(runner->model_params(model_id));
        m_num_states += params.count;
    }

    ModBaseInfo mod_info = config::get_modbase_info(base_mod_params);
    m_mod_base_info = std::make_shared<ModBaseInfo>(
            std::move(mod_info.alphabet), std::move(mod_info.long_names), context_handler.encode());
    m_base_prob_offsets = mod_info.base_probs_offsets();
}

void ModBaseChunkCallerNode::validate_runners() const {
    for (const auto& runner : m_runners) {
        if (!runner->takes_chunk_inputs()) {
            throw std::logic_error("Modbase model runner invalid for chunked caller: '" +
                                   runner->get_name() + "'.");
        }
    }

    for (size_t model_id = 0; model_id < m_runners.at(0)->num_models(); ++model_id) {
        const auto& config = m_runners.at(0)->model_params(model_id);

        std::string name = "Modbase '";
        name.push_back(config.mods.base);
        name += "' config error - ";
        if (!config.is_chunked_input_model()) {
            throw std::runtime_error(name + "Runner requires chunked input modbase model.");
        }

        const auto& ctx = config.context;
        if (ctx.chunk_size < ctx.samples_before + ctx.samples_after) {
            spdlog::error("{} chunk_size:{} is less than samples_before:{} + samples_after:{}.",
                          name, ctx.chunk_size, ctx.samples_before, ctx.samples_after);
            throw std::runtime_error(name + "chunk_size is smaller than the context window.");
        }

        if (ctx.chunk_size % m_canonical_stride != 0) {
            spdlog::error("{} chunk_size:{} is not divisible by canonical_model_stride:{}.", name,
                          ctx.chunk_size, m_canonical_stride);
            throw std::runtime_error(name + "model strides are not compatible.");
        }

        if (m_canonical_stride % config.general.stride != 0) {
            spdlog::error("{} canonical_model_stride:{} is not divisible by modbase_stride:{}.",
                          name, m_canonical_stride, config.general.stride);
            throw std::runtime_error(name + "model strides are not compatible.");
        }

        if (ctx.chunk_size % config.general.stride != 0) {
            spdlog::error("{} chunk_size:{} is not divisible by modbase_stride:{}.", name,
                          ctx.chunk_size, config.general.stride);
            throw std::runtime_error(name + "model stride and chunk size are not compatible.");
        }

        if (ctx.kmer_len != m_kmer_len) {
            throw std::runtime_error(name + "all models must use the same kmer length.");
        }

        if (ctx.reverse != m_is_rna_model) {
            throw std::runtime_error(name + "all models must be exclusively for DNA or RNA.");
        }
    }
}

void ModBaseChunkCallerNode::initialise_base_mod_probs(ReadCommon& read) const {
    // initialize base_mod_probs _before_ we start handing out chunks
    read.base_mod_probs.resize(read.seq.size() * m_num_states, 0);
    for (size_t i = 0; i < read.seq.size(); ++i) {
        // Initialize for what corresponds to 100% canonical base for each position.
        // This is like one-hot encoding the canonical bases
        int base_id = utils::BaseInfo::BASE_IDS.at(read.seq[i]);
        if (base_id < 0) {
            spdlog::error("Modbase input failed - invalid character - seq[{}]='{}' id:{}.", i,
                          read.seq[i], read.read_id);
            throw std::runtime_error("Invalid character in sequence.");
        }
        read.base_mod_probs[i * m_num_states + m_base_prob_offsets.at(base_id)] = 1;
    }
    read.mod_base_info = m_mod_base_info;
}

// Get the index of the next context hit in `hit_sig_idxs` with a signal index
// greater than or equal to `chunk_signal_start`.
std::optional<int64_t> ModBaseChunkCallerNode::next_hit(const std::vector<int64_t>& hit_sig_idxs,
                                                        const int64_t chunk_signal_start) {
    // Check for the first element explicitly
    if (!hit_sig_idxs.empty() && hit_sig_idxs.front() >= chunk_signal_start) {
        return 0;
    }

    // The first context hit signal index at or after `chunk_signal_start`
    const auto next_hit =
            std::lower_bound(hit_sig_idxs.begin(), hit_sig_idxs.end(), chunk_signal_start);

    if (next_hit != hit_sig_idxs.cend()) {
        return std::distance(hit_sig_idxs.cbegin(), next_hit);
    }

    // Did not find a context hit in this chunk or any remaining chunk
    return std::nullopt;
}

std::vector<uint64_t> ModBaseChunkCallerNode::get_seq_to_sig_map(const std::vector<uint8_t>& moves,
                                                                 const size_t signal_len,
                                                                 const size_t reserve) const {
    nvtx3::scoped_range range{"pop_s2s_map"};
    auto seq_to_sig_map = utils::moves_to_map(moves, m_canonical_stride, signal_len, reserve);
    if (m_is_rna_model) {
        assert(signal_len % m_canonical_stride == 0);
        utils::reverse_seq_to_sig_map(seq_to_sig_map, signal_len);
    }
    return seq_to_sig_map;
}

// Populate the hits for each canonical base if it's needed by a caller.
bool ModBaseChunkCallerNode::populate_hits_seq(PerBaseIntVec& per_base_hits_seq,
                                               const std::string& seq,
                                               const modbase::RunnerPtr& runner) const {
    nvtx3::scoped_range range{"pop_hits_seq"};
    bool has_hits = false;
    for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
        const std::vector<size_t> motif_hits = runner->get_motif_hits(model_id, seq);
        auto& hits_seq = per_base_hits_seq.at(runner->model_params(model_id).mods.base_id);
        hits_seq.resize(motif_hits.size());

        for (size_t i = 0; i < motif_hits.size(); ++i) {
            hits_seq[i] = static_cast<int64_t>(motif_hits[i]);
        }

        has_hits |= !hits_seq.empty();
    }
    return has_hits;
}

// Translate the sequence-space hits into signal-space
void ModBaseChunkCallerNode::populate_hits_sig(PerBaseIntVec& per_base_hits_sig,
                                               const PerBaseIntVec& per_base_hits_seq,
                                               const std::vector<uint64_t>& seq_to_sig_map) const {
    nvtx3::scoped_range range{"pop_hits_sig"};
    const auto& runner = m_runners.at(0);
    for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
        const int base_id = runner->model_params(model_id).mods.base_id;
        const auto& hits_seq = per_base_hits_seq.at(base_id);
        auto& hits_sig = per_base_hits_sig.at(base_id);

        hits_sig.resize(hits_seq.size());
        for (size_t i = 0; i < hits_seq.size(); ++i) {
            hits_sig[i] = seq_to_sig_map.at(hits_seq[i]);
        }
    }
}

void ModBaseChunkCallerNode::populate_encoded_kmer(
        std::vector<int8_t>& encoded_kmer,
        const size_t output_size,
        const std::vector<int>& int_seq,
        const std::vector<uint64_t>& seq_to_sig_map) const {
    nvtx3::scoped_range range{"pop_enc_kmer"};
    // Center the kmer over the "focus" base instead of placing it first
    const bool kIsCentered = utils::get_dev_opt<bool>("kmer_centered", true);
    encoded_kmer = modbase::encode_kmer_chunk(int_seq, seq_to_sig_map, m_kmer_len, output_size, 0,
                                              kIsCentered);
}

void ModBaseChunkCallerNode::populate_signal(at::Tensor& signal,
                                             std::vector<uint64_t>& seq_to_sig_map,
                                             const at::Tensor& raw_data,
                                             const std::vector<int>& int_seq,
                                             const modbase::RunnerPtr& runner) const {
    if (m_is_rna_model) {
        nvtx3::scoped_range range{"pop_sig_rna"};

        // Reverse the RNA signal and prepend a short mirrored slice of padding to ensure moves are
        // stride aligned.
        const int64_t len = raw_data.size(0);
        const int64_t padding = utils::pad_to(len, m_canonical_stride) - len;

#if TORCH_VERSION_MAJOR < 2
        at::Tensor sig =
                at::concat({raw_data.slice(0, len - padding, len), at::flip(raw_data, 0)}, 0);
#else
        at::Tensor sig = at::empty({len + padding}, raw_data.options());
        at::Tensor body = sig.slice(0, padding, len + padding);
        at::flip_out(body, raw_data, 0);

        sig.slice(0, 0, padding) = raw_data.slice(0, len - padding, len);
#endif

        signal = runner->scale_signal(0, sig, int_seq, seq_to_sig_map);
        return;
    }
    nvtx3::scoped_range range{"pop_sig"};
    signal = runner->scale_signal(0, raw_data, int_seq, seq_to_sig_map);
    return;
}

// For each caller, get chunk definitions which always contain a context hit.
std::vector<ModBaseChunkCallerNode::ModBaseChunks> ModBaseChunkCallerNode::get_chunks(
        const modbase::RunnerPtr& runner,
        const std::shared_ptr<WorkingRead>& working_read,
        const bool is_template) const {
    nvtx3::scoped_range range{"mbc_get_chunks"};
    std::vector<ModBaseChunks> chunks_to_enqueue_by_model(runner->num_models());

    // Complement modbase data is stored in the template direction so there's no need for
    // any indexing gymnastics in chunk creation
    auto modbase_data = is_template ? working_read->template_data : working_read->complement_data;

    const int64_t signal_len = modbase_data.signal.size(0);

    for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
        auto& chunks_to_enqueue = chunks_to_enqueue_by_model.at(model_id);

        const auto& config = runner->model_params(model_id);
        const auto base_id = config.mods.base_id;

        const auto& hits_to_sig = modbase_data.per_base_hits_sig[base_id];

        const auto num_states = config.mods.count + 1;
        const auto& ctx = config.context;
        const auto chunk_starts = ModBaseChunkCallerNode::get_chunk_starts(
                signal_len, hits_to_sig, ctx.chunk_size, ctx.samples_before, ctx.samples_after,
                m_pad_end_align);

        for (const auto& [chunk_start, hit_idx] : chunk_starts) {
            chunks_to_enqueue.emplace_back(
                    std::make_unique<ModBaseChunk>(working_read, int(model_id), base_id,
                                                   chunk_start, hit_idx, num_states, is_template));
        }
        working_read->num_modbase_chunks += chunk_starts.size();
    }
    return chunks_to_enqueue_by_model;
}

std::vector<std::pair<int64_t, int64_t>> ModBaseChunkCallerNode::get_chunk_starts(
        const int64_t signal_len,
        const std::vector<int64_t>& hits_to_sig,
        const int64_t chunk_size,
        const int64_t context_samples_before,
        const int64_t context_samples_after,
        const bool end_align_last_chunk) {
    std::vector<std::pair<int64_t, int64_t>> chunks;

    int64_t chunk_st = 0;  ///< chunk start in signal-space
    while (chunk_st < signal_len) {
        std::optional<int64_t> next_hit = ModBaseChunkCallerNode::next_hit(hits_to_sig, chunk_st);

        if (!next_hit.has_value()) {
            break;
        }

        const int64_t hit_idx = next_hit.value();
        const int64_t hit_sig = hits_to_sig.at(hit_idx);

        // Add context samples as a lead-in
        chunk_st = hit_sig - context_samples_before;
        // If there's no lead-in context start at the first sample
        chunk_st = chunk_st > 0 ? chunk_st : 0;
        chunks.emplace_back(std::make_pair(chunk_st, hit_idx));

        // Step chunk forward. Ensure hits with incomplete downstream context are not skipped
        chunk_st += chunk_size - context_samples_after + 1;
        // Always move forward if chunk_size = before+after
        if (chunk_st <= hit_sig) {
            chunk_st = hit_sig + 1;
        }
    }

    if (chunks.size() > 1 && end_align_last_chunk) {
        const int64_t last_hit = hits_to_sig.back();
        const int64_t aligned_chunk_st = last_hit + context_samples_after - chunk_size;
        if (aligned_chunk_st > 0) {
            chunks.back().first = aligned_chunk_st;
        }
    }

    return chunks;
}

template <typename ReadType>
void ModBaseChunkCallerNode::add_read_to_working_set(std::unique_ptr<ReadType> read_ptr,
                                                     std::shared_ptr<WorkingRead> working_read) {
    if (!read_ptr || !working_read) {
        throw std::invalid_argument("Null pointer passed to add_read_to_working_set.");
    }

    // Hand over our ownership to the working read
    working_read->read = std::move(read_ptr);

    // Put the working read in the working read set, where it will remain until its
    // chunks are all called.
    {
        std::lock_guard<std::mutex> working_reads_lock(m_working_reads_mutex);
        m_working_reads.insert(std::move(working_read));
    }
}

bool ModBaseChunkCallerNode::populate_modbase_data(ModBaseData& mbd,
                                                   const modbase::RunnerPtr& runner,
                                                   const std::string& seq,
                                                   const at::Tensor& signal,
                                                   const std::vector<uint8_t>& moves,
                                                   const std::string& read_id) const {
    // For RNA: Pad signal length to be evenly divisible by the canonical stride so that the
    // sequence to signal mapping is always stride aligned and not offset by any remainder
    // in the last move (which becomes the first move when reversed).
    const size_t signal_len =
            m_is_rna_model ? utils::pad_to(signal.size(0), m_canonical_stride) : signal.size(0);

    if (!populate_hits_seq(mbd.per_base_hits_seq, seq, runner)) {
        return false;
    }

    std::vector<uint64_t> seq_to_sig_map = get_seq_to_sig_map(moves, signal_len, seq.size() + 1);
    std::vector<int> int_seq = utils::sequence_to_ints(seq);

    populate_hits_sig(mbd.per_base_hits_sig, mbd.per_base_hits_seq, seq_to_sig_map);
    populate_signal(mbd.signal, seq_to_sig_map, signal, int_seq, runner);
    populate_encoded_kmer(mbd.encoded_kmers, signal_len, int_seq, seq_to_sig_map);

    if (signal_len != static_cast<size_t>(mbd.signal.size(0))) {
        spdlog::error("Modbase signal length is incorrect for read:'{}' - {}!={}", read_id,
                      signal_len, mbd.signal.size(0));
        throw std::runtime_error("Modbase signal processing failed.");
    }

    const auto enc_kmer_size = mbd.encoded_kmers.size();
    const auto expected = signal_len * m_kmer_len * utils::BaseInfo::NUM_BASES;
    if (enc_kmer_size != expected) {
        spdlog::error("Modbase kmer encoding failed for read:'{}' - {}!={}", read_id, enc_kmer_size,
                      expected);
        throw std::runtime_error("Modbase kmer encoding failed.");
    }

    return true;
}

void ModBaseChunkCallerNode::simplex_mod_call(Message&& message) {
    // The callers of each runner are all the same, so just take the first one.
    auto& runner = m_runners.at(0);

    // Get ownership of the read
    auto read_ptr = std::get<SimplexReadPtr>(std::move(message));
    auto& read = read_ptr->read_common;
    const std::string read_id = read.read_id;

    stats::Timer timer;

    initialise_base_mod_probs(read);

    auto working_read = std::make_shared<WorkingRead>();
    auto& modbase_data = working_read->template_data;

    if (!populate_modbase_data(modbase_data, runner, read.seq, read.raw_data, read.moves,
                               read_id)) {
        send_message_to_sink(std::move(read_ptr));
        return;
    }

    constexpr bool kIsTemplate = true;
    std::vector<ModBaseChunks> chunks_by_caller = get_chunks(runner, working_read, kIsTemplate);

    add_read_to_working_set(std::move(read_ptr), std::move(working_read));

    // Push the chunks to the chunk queues.
    // Needs to be done after working_read->read is set as chunks could be processed
    // before we set that value otherwise.
    for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
        auto& chunk_queue = m_chunk_queues.at(model_id);
        auto& chunks_to_enqueue = chunks_by_caller.at(model_id);
        for (auto& chunk : chunks_to_enqueue) {
            chunk_queue->try_push(std::move(chunk));
        }
    }
}

void ModBaseChunkCallerNode::duplex_mod_call(Message&& message) {
    // The callers of each runner are all the same, so just take the first one.
    auto& runner = m_runners.at(0);

    // Get ownership of the read
    auto read_ptr = std::get<DuplexReadPtr>(std::move(message));
    auto& read_common = read_ptr->read_common;
    auto& read_stereo = read_ptr->stereo_feature_inputs;
    const std::string read_id = read_common.read_id;

    stats::Timer timer;

    initialise_base_mod_probs(read_common);

    std::vector<ModBaseChunks> chunks_by_caller_template;
    std::vector<ModBaseChunks> chunks_by_caller_complement;

    auto working_read = std::make_shared<WorkingRead>();
    working_read->is_duplex = true;

    try {
        for (const bool is_template_direction : {true, false}) {
            auto& modbase_data = is_template_direction ? working_read->template_data
                                                       : working_read->complement_data;

            // ??? Why are we flipping the signal but not the moves?
            auto simplex_signal = is_template_direction
                                          ? read_stereo.template_signal
                                          : at::flip(read_stereo.complement_signal, 0);

            // const-ref extends lifetime of temporary
            const auto& simplex_moves = is_template_direction ? read_stereo.template_moves
                                                              : read_stereo.complement_moves;

            // const-ref extends lifetime of temporary
            const auto& simplex_seq =
                    is_template_direction ? read_stereo.template_seq
                                          : utils::reverse_complement(read_stereo.complement_seq);

            // const-ref extends lifetime of temporary
            const auto& duplex_seq = is_template_direction
                                             ? read_common.seq
                                             : utils::reverse_complement(read_common.seq);

            auto [moves_offset, target_start, new_move_table] =
                    utils::realign_moves(simplex_seq, duplex_seq, simplex_moves);

            // If the alignment has failed, the rest of this duplex mod call cannot be
            // completed in this direction
            if (moves_offset == -1 && target_start == -1 && new_move_table.empty()) {
                continue;
            }

            // Store the realigned target start to correctly resolve the sequence offset
            // in the output predictions back onto the original sequence.
            modbase_data.target_start = target_start;

            auto new_signal_len = new_move_table.size() * m_canonical_stride;
            auto new_num_moves = std::accumulate(new_move_table.begin(), new_move_table.end(), 0);
            auto new_seq = duplex_seq.substr(target_start, new_num_moves);
            auto signal = simplex_signal.slice(0, moves_offset * m_canonical_stride,
                                               moves_offset * m_canonical_stride + new_signal_len);

            if (!populate_modbase_data(modbase_data, runner, new_seq, signal, new_move_table,
                                       read_id)) {
                continue;
            }

            auto& chunks_by_caller =
                    is_template_direction ? chunks_by_caller_template : chunks_by_caller_complement;
            chunks_by_caller = get_chunks(runner, working_read, is_template_direction);
        }
    } catch (const std::exception& e) {
        spdlog::error("ModBase Duplex Caller: {}", e.what());
    }

    if (chunks_by_caller_template.empty() && chunks_by_caller_complement.empty()) {
        send_message_to_sink(std::move(read_ptr));
        return;
    }

    add_read_to_working_set(std::move(read_ptr), std::move(working_read));

    // Push the chunks to the chunk queues.
    // Needs to be done after working_read->read is set as chunks could be processed
    // before we set that value otherwise.
    for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
        auto& chunk_queue = m_chunk_queues.at(model_id);
        if (!chunks_by_caller_template.empty()) {
            for (auto& template_chunk : chunks_by_caller_template.at(model_id)) {
                chunk_queue->try_push(std::move(template_chunk));
            }
        }
        if (!chunks_by_caller_complement.empty()) {
            for (auto& complement_chunk : chunks_by_caller_complement.at(model_id)) {
                chunk_queue->try_push(std::move(complement_chunk));
            }
        }
    }
}

void ModBaseChunkCallerNode::create_and_submit_chunks(
        modbase::RunnerPtr& runner,
        const size_t model_id,
        const int64_t previous_chunk_count,
        std::vector<std::unique_ptr<ModBaseChunk>>& batched_chunks) const {
    const auto& cfg = runner->model_params(model_id);
    const int64_t chunk_size = cfg.context.chunk_size;
    const int kmer_size_per_sample = cfg.context.kmer_len * utils::BaseInfo::NUM_BASES;

    const int64_t num_chunks = static_cast<int64_t>(batched_chunks.size());
    // We have just grabbed a number of chunks (0 in the case of timeout) from
    // the chunk queue and added them to batched_chunks.  Insert those chunks
    // into the model input tensors.
    for (int64_t chunk_idx = previous_chunk_count; chunk_idx < num_chunks; ++chunk_idx) {
        assert(chunk_idx < m_batch_size);
        const auto& chunk = batched_chunks.at(chunk_idx);

        const auto& working_read = chunk->working_read;
        const auto& modbase_data =
                chunk->is_template ? working_read->template_data : working_read->complement_data;
        const auto& read_id = get_read_common_data(working_read->read).read_id;

        const int64_t start = chunk->signal_start;
        const int64_t end =
                std::min(start + chunk_size, static_cast<int64_t>(modbase_data.signal.size(0)));
        const int64_t len = end - start;

        if (start > end) {
            spdlog::error("Modbase chunking failed - start>end {}>{} id:{}.", start, end, read_id);
            throw std::runtime_error("Modbase chunking failed.");
        }
        if (int64_t(modbase_data.encoded_kmers.size()) < end) {
            spdlog::error("Modbase chunking failed - out of bounds {}<{} id:{}.",
                          modbase_data.encoded_kmers.size(), end, read_id);
            throw std::runtime_error("Modbase chunking failed.");
        }

        auto signal_chunk = modbase_data.signal.index({at::indexing::Slice(start, end)});
        // TODO -- this copying could be eliminated by writing directly into the runner input_seqs_ptr
        auto encoded_kmers_chunk = std::vector<int8_t>(
                modbase_data.encoded_kmers.begin() + start * kmer_size_per_sample,
                modbase_data.encoded_kmers.begin() + end * kmer_size_per_sample);

        // Add any necessary padding to the end of the chunk.
        if (len < chunk_size) {
            // Tile the signal tensor
            auto [n_tiles, overhang] = std::div(chunk_size, len);
            signal_chunk = at::concat({signal_chunk.repeat({n_tiles}),
                                       signal_chunk.index({at::indexing::Slice(0, overhang)})},
                                      -1);
            // Tile the kmer vector
            const int64_t original_size = static_cast<int64_t>(encoded_kmers_chunk.size());
            const int64_t extended_size = chunk_size * kmer_size_per_sample;
            encoded_kmers_chunk.resize(extended_size);

            for (int64_t i = original_size; i < extended_size; ++i) {
                encoded_kmers_chunk[i] = encoded_kmers_chunk[i % original_size];
            }
        }

        runner->accept_chunk(static_cast<int>(model_id), static_cast<int>(chunk_idx), signal_chunk,
                             encoded_kmers_chunk);
    }
}
void ModBaseChunkCallerNode::chunk_caller_thread_fn(const size_t worker_id, const size_t model_id) {
    utils::set_thread_name("mbc_chunk_caller");
    at::InferenceMode inference_mode_guard;

    auto& runner = m_runners.at(worker_id);
    auto& chunk_queue = m_chunk_queues.at(model_id);

    std::vector<std::unique_ptr<ModBaseChunk>> batched_chunks;
    auto last_chunk_reserve_time = std::chrono::system_clock::now();

    int64_t previous_chunk_count = 0;

    while (true) {
        nvtx3::scoped_range range{"chunk_caller_thread_fn"};
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

        create_and_submit_chunks(runner, model_id, previous_chunk_count, batched_chunks);

        // If we have a complete batch, or we have a partial batch and timed out,
        // then call what we have.
        if (batched_chunks.size() == size_t(m_batch_size) ||
            (status == utils::AsyncQueueStatus::Timeout && !batched_chunks.empty())) {
            if (batched_chunks.size() != size_t(m_batch_size)) {
                ++m_num_partial_batches_called;
            }
            // Input tensor is full, let's get scores.
            call_batch(worker_id, model_id, batched_chunks);
        }

        previous_chunk_count = batched_chunks.size();
    }

    // Basecall any remaining chunks.
    if (!batched_chunks.empty()) {
        call_batch(worker_id, model_id, batched_chunks);
    }

    // Reduce the count of active model callers.  If this was the last active
    // model caller also send termination signal to sink
    int num_remaining_callers = --m_num_active_runner_workers;
    if (num_remaining_callers == 0) {
        m_processed_chunks.terminate();
    }
}

void ModBaseChunkCallerNode::call_batch(
        const size_t worker_id,
        const size_t model_id,
        std::vector<std::unique_ptr<ModBaseChunk>>& batched_chunks) {
    nvtx3::scoped_range loop{"call_batch"};

    dorado::stats::Timer timer;
    // Results shape (N, strides*preds)
    auto results = m_runners.at(worker_id)->call_chunks(static_cast<int>(model_id),
                                                        static_cast<int>(batched_chunks.size()));

    // Convert results to float32 with one call and address via a raw pointer,
    // to avoid huge libtorch indexing overhead.
    auto results_f32 = results.to(at::ScalarType::Float);

    assert(results_f32.is_contiguous());
    const auto* const results_f32_ptr = results_f32.data_ptr<float>();
    const auto row_size = results.size(1);

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

int64_t ModBaseChunkCallerNode::resolve_score_index(const int64_t hit_sig_abs,
                                                    const int64_t chunk_signal_start,
                                                    const int64_t scores_states,
                                                    const int64_t chunk_size,
                                                    const int64_t context_samples_before,
                                                    const int64_t context_samples_after,
                                                    const int64_t modbase_stride) {
    if (hit_sig_abs < chunk_signal_start) {
        spdlog::error(
                "hit_sig_abs:{} chunk_signal_start:{} scores_states:{} chunk_size:{} "
                "context_samples_before:{} context_samples_after:{} modbase_stride:{}",
                hit_sig_abs, chunk_signal_start, scores_states, chunk_size, context_samples_before,
                context_samples_after, modbase_stride);
        spdlog::error("Modbase hit before chunk start: {}<{}", hit_sig_abs, chunk_signal_start);
        throw std::runtime_error("Modbase hit before chunk start.");
    }

    // Context hit chunk-relative signal index
    const int64_t hit_sig_rel = hit_sig_abs - chunk_signal_start;

    // Skip hits at end of a chunk without enough downstream context
    // It will be processed at the start if the next chunk with complete context
    if (hit_sig_rel > chunk_size - context_samples_after) {
        return -2;
    }

    // Skip hits at the start of a chunk with insufficient context
    // This hit will have been processed in a previous chunk
    // UNLESS it's the start of a read where there's no useful lead-in
    if (hit_sig_abs > context_samples_before && hit_sig_rel < context_samples_before) {
        return -1;
    }

    // We should land on a canonical base
    if (hit_sig_rel % modbase_stride != 0) {
        spdlog::error(
                "Modbase unaligned chunk - hit_absolute:{} hit_relative:{} stride:{} "
                "chunk_start:{}",
                hit_sig_abs, hit_sig_rel, modbase_stride, chunk_signal_start);
        throw std::runtime_error("Modbase score did not align to canonical base.");
    }

    // Convert chunk-relative signal-space score index into sequence-space (/stride)
    // and then into scores-space (*num_states)
    return hit_sig_rel / modbase_stride * scores_states;
}

int64_t ModBaseChunkCallerNode::resolve_duplex_sequence_index(const int64_t hit,
                                                              const int64_t target_start,
                                                              const int64_t sequence_size,
                                                              const bool is_template_direction) {
    if (is_template_direction) {
        return hit + target_start;
    }
    return sequence_size - (hit + target_start + 1);
}

void ModBaseChunkCallerNode::output_thread_fn() {
    at::InferenceMode inference_mode_guard;
    utils::set_thread_name("mbc_output");
    const auto& runner = m_runners.at(0);

    std::unique_ptr<ModBaseChunk> chunk;
    while (m_processed_chunks.try_pop(chunk) == utils::AsyncQueueStatus::Success) {
        auto working_read = chunk->working_read;
        auto& read = get_read_common_data(working_read->read);

        // Extract useful modbase model parameters
        const auto& cfg = runner->model_params(chunk->model_id);
        const char modbase_model_base = cfg.mods.base;
        const int64_t modbase_stride = cfg.general.stride;
        const int64_t chunk_size = cfg.context.chunk_size;
        const int64_t context_samples_before = cfg.context.samples_before;
        const int64_t context_samples_after = cfg.context.samples_after;

        // The number of states predicted by this modbase model `num_mods + 1`
        const int64_t scores_states = chunk->num_states;
        const int64_t scores_size = static_cast<int64_t>(chunk->scores.size());
        const int64_t scores_seq_len = scores_size / scores_states;

        if (scores_seq_len > chunk_size / modbase_stride) {
            spdlog::error("Modbase scores too long {}>{}/{}", scores_seq_len, chunk_size,
                          modbase_stride);
            throw std::runtime_error("Modbase scores too long.");
        }

        const bool is_template_direction = chunk->is_template;
        const ModBaseData& modbase_data =
                is_template_direction ? working_read->template_data : working_read->complement_data;

        const std::vector<int64_t>& hits_seq = modbase_data.per_base_hits_seq.at(chunk->base_id);
        const std::vector<int64_t>& hits_sig = modbase_data.per_base_hits_sig.at(chunk->base_id);

        // The offset into the mod probs for the canonical base
        const int64_t base_offset = static_cast<int64_t>(m_base_prob_offsets.at(cfg.mods.base_id));

        for (size_t hit = chunk->hit_start; hit < hits_sig.size(); ++hit) {
            // Context hit sequence index in the chunk sequence
            const int64_t hit_seq = !working_read->is_duplex
                                            ? hits_seq.at(hit)
                                            : resolve_duplex_sequence_index(
                                                      hits_seq.at(hit), modbase_data.target_start,
                                                      read.seq.size(), is_template_direction);

            const auto& seq = is_template_direction
                                      ? read.seq[hit_seq]
                                      : dorado::utils::complement_table[read.seq[hit_seq]];

            // The canonical base should be constant for a single model
            if (seq != modbase_model_base) {
                spdlog::error(
                        "Modbase base '{}' from hit index '{}' does not match the model "
                        "context '{}' on '{}' {}",
                        seq, hit, modbase_model_base, read.read_id, is_template_direction);
                throw std::runtime_error("Modbase hit base is not correct.");
            }

            int64_t hit_score_idx = resolve_score_index(
                    hits_sig.at(hit), chunk->signal_start, scores_states, chunk_size,
                    context_samples_before, context_samples_after, modbase_stride);

            if (hit_score_idx <= -2) {
                // No more hits in this chunk
                break;
            } else if (hit_score_idx == -1) {
                // This hit is skipped
                continue;
            }

            // Extract the scores for the canonical base and each of the mods in this model
            for (int64_t mod_offset = 0; mod_offset < scores_states; ++mod_offset) {
                const int64_t score_idx = hit_score_idx + mod_offset;
                if (score_idx >= scores_size) {
                    spdlog::error("Modbase score index out of bounds: {}>={} on '{}", score_idx,
                                  scores_size, read.read_id);
                    throw std::runtime_error("Modbase score index out of bounds.");
                }

                const uint8_t score = static_cast<uint8_t>(
                        std::min(std::floor(chunk->scores[score_idx] * 256), 255.0f));

                // Index into the probabilities is calculated by
                // sequence_index * num_states := canonical "A" base probs index
                // offset then by the canonical base modification offsets
                const int64_t prob_idx = hit_seq * m_num_states + base_offset + mod_offset;
                read.base_mod_probs.at(prob_idx) = score;
            }
        }

        // If all chunks for the read associated with this chunk have now been called,
        // send it on to the sink and erase the working read.
        auto num_chunks_called = ++working_read->num_modbase_chunks_called;
        if (num_chunks_called == working_read->num_modbase_chunks) {
            m_num_samples_processed_incl_padding += num_chunks_called * chunk_size;

            // Send the completed read on to the sink.
            send_message_to_sink(std::move(working_read->read));

            // Remove it from the working read set.
            std::lock_guard<std::mutex> working_reads_lock(m_working_reads_mutex);
            auto read_iter = m_working_reads.find(working_read);
            if (read_iter != m_working_reads.end()) {
                m_working_reads.erase(read_iter);
            } else {
                auto read_id = get_read_common_data(working_read->read).read_id;
                throw std::runtime_error("Modbase expected to find read id " + read_id +
                                         " in working reads set but it doesn't exist.");
            }
        }
    }
}

std::unordered_map<std::string, double> ModBaseChunkCallerNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    for (const auto& runner : m_runners) {
        const auto runner_stats = stats::from_obj(*runner);
        stats.insert(runner_stats.begin(), runner_stats.end());
    }
    stats["batches_called"] = double(m_num_batches_called);
    stats["partial_batches_called"] = double(m_num_partial_batches_called);
    stats["samples_incl_padding"] = double(m_num_samples_processed_incl_padding);
    return stats;
}

}  // namespace dorado
