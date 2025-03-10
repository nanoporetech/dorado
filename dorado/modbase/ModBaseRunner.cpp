#include "ModBaseRunner.h"

#include "ModBaseCaller.h"
#include "ModbaseScaler.h"
#include "config/ModBaseModelConfig.h"
#include "torch_utils/tensor_utils.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#endif

#include <torch/torch.h>

namespace {
#if DORADO_CUDA_BUILD
std::vector<c10::optional<c10::Stream>> get_streams_from_caller(
        const std::shared_ptr<dorado::modbase::ModBaseCaller>& caller) {
    std::vector<c10::optional<c10::Stream>> streams;
    for (size_t i = 0; i < caller->num_models(); ++i) {
        if (caller->device().is_cuda()) {
            streams.push_back(c10::cuda::getStreamFromPool(false, caller->device().index()));
        } else {
            streams.emplace_back();
        }
    }
    return streams;
}
#endif
}  // namespace

namespace dorado::modbase {

ModBaseRunner::ModBaseRunner(std::shared_ptr<ModBaseCaller> caller)
        : m_caller(std::move(caller)),
          m_input_sigs(m_caller->create_input_sig_tensors()),
          m_input_seqs(m_caller->create_input_seq_tensors()),
          m_is_chunked_model_type(is_chunked_model_type())
#if DORADO_CUDA_BUILD
          ,
          m_streams(get_streams_from_caller(m_caller))
#endif
{
}

void ModBaseRunner::accept_chunk(int model_id,
                                 int chunk_idx,
                                 const at::Tensor& signal,
                                 const std::vector<int8_t>& kmers) {
    // As usual, avoid torch indexing because it is glacially slow.
    // GPU base calling uses float16 signals and input tensors.
    // CPU base calling uses float16 signals, float32 input tensors.
    // Both versions take int8 sequence encodings.

    auto& input_sigs = m_input_sigs[model_id];
    auto& input_seqs = m_input_seqs[model_id];
    if (signal.size(0) != input_sigs.size(2)) {
        throw std::logic_error(
                "ModBaseRunner received signal and sequence chunks with different lengths.");
    }
    if (!input_sigs.is_contiguous()) {
        spdlog::warn(
                "ModBaseRunner::accept_chunk received non-contiguous signal tensor which will "
                "impact performance.");
        input_sigs = input_sigs.contiguous();
    }
    if (!input_seqs.is_contiguous()) {
        spdlog::warn(
                "ModBaseRunner::accept_chunk received non-contiguous sequence tensor which will "
                "impact performance.");
        input_seqs = input_seqs.contiguous();
    }

    const auto sig_len = signal.size(0);
    dorado::utils::copy_tensor_elems(input_sigs, chunk_idx * sig_len, signal, 0, sig_len);

    const auto kmer_elem_count = input_seqs.size(1) * input_seqs.size(2);
    if (input_seqs.dtype() != torch::kInt8) {
        throw std::runtime_error("ModBaseRunner has unsupported input sequence dtype");
    }
    using SeqInputType = int8_t;
    SeqInputType* const input_seqs_ptr = input_seqs.data_ptr<SeqInputType>();
    std::memcpy(&input_seqs_ptr[chunk_idx * kmer_elem_count], kmers.data(),
                kmer_elem_count * sizeof(SeqInputType));
}

at::Tensor ModBaseRunner::call_chunks(int model_id, int num_chunks) {
#if DORADO_CUDA_BUILD
    c10::cuda::OptionalCUDAStreamGuard guard(m_streams[model_id]);
#endif
    return m_caller->call_chunks(model_id, m_input_sigs[model_id], m_input_seqs[model_id],
                                 num_chunks);
}

at::Tensor ModBaseRunner::scale_signal(size_t model_id,
                                       at::Tensor signal,
                                       const std::vector<int>& seq_ints,
                                       const std::vector<uint64_t>& seq_to_sig_map) const {
    auto& scaler = m_caller->modbase_model_data(model_id)->scaler;
    if (scaler) {
        return scaler->scale_signal(signal, seq_ints, seq_to_sig_map);
    }
    return signal;
}

std::vector<size_t> ModBaseRunner::get_motif_hits(size_t model_id, const std::string& seq) const {
    return m_caller->modbase_model_data(model_id)->get_motif_hits(seq);
}

const config::ModBaseModelConfig& ModBaseRunner::model_params(size_t model_id) const {
    return m_caller->modbase_model_data(model_id)->params;
}

size_t ModBaseRunner::num_models() const { return m_caller->num_models(); }

void ModBaseRunner::terminate() { m_caller->terminate(); }
void ModBaseRunner::restart() { m_caller->restart(); }

bool ModBaseRunner::is_chunked_model_type() const {
    // Assert all models are either chunked or context-centered.
    const bool is_chunked_model = model_params(0).is_chunked_input_model();
    for (size_t i = 1; i < num_models(); i++) {
        if (is_chunked_model != model_params(i).is_chunked_input_model()) {
            const auto base_0 = std::string(1, model_params(0).mods.base);
            const auto base_i = std::string(1, model_params(i).mods.base);

            const auto type_0 = to_string(model_params(0).general.model_type);
            const auto type_i = to_string(model_params(i).general.model_type);

            spdlog::error("Modbase models have different types {}:{} {}:{}.", base_0, type_0,
                          base_i, type_i);
            throw std::runtime_error("Cannot mix the types of modified bases models");
        }
    }
    return is_chunked_model;
}

bool ModBaseRunner::takes_chunk_inputs() const { return m_is_chunked_model_type; }

std::string ModBaseRunner::get_name() const {
    std::ostringstream name_stream;
    name_stream << "ModBaseRunner_" << this;
    return name_stream.str();
}

stats::NamedStats ModBaseRunner::sample_stats() const {
    // We don't have direct access to the caller object when the pipeline is set up,
    // so pass through stats here.
    // Each runner will retrieve stats from the caller.
    // Only the last retrieved version will appear, but they should be very similar.
    stats::NamedStats stats = stats::from_obj(*m_caller);
    stats["batches_called"] = double(m_num_batches_called);
    return stats;
}

}  // namespace dorado::modbase
