#include "decoder_base.h"

#include "torch_utils/tensor_utils.h"
#include "utils/span.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace dorado::secondary {

DecoderBase::DecoderBase(const LabelSchemeType label_scheme_type)
        : m_label_scheme_type(label_scheme_type) {}

std::vector<std::vector<secondary::ConsensusResult>> DecoderBase::decode_bases(
        const at::Tensor& logits) const {
    return decode_batch_bases_impl(m_label_scheme_type, logits);
}

std::string DecoderBase::get_label_scheme_symbols() const {
    return label_scheme_symbols(m_label_scheme_type);
}

LabelSchemeType parse_label_scheme_type(const std::string& type) {
    if (type == "HaploidLabelScheme") {
        return LabelSchemeType::HAPLOID;
    }
    throw std::runtime_error{"Unknown label scheme type: '" + type + "'!"};
}

std::vector<std::vector<secondary::ConsensusResult>> decode_batch_bases_impl(
        const std::string& symbols,
        const dorado::Span<const float> logits,
        const size_t batch_size,
        const size_t sequence_length,
        const size_t num_haplotypes,
        const size_t num_classes) {
    constexpr double QV_CAP = 70.0;
    const double min_err = std::pow(10.0, -QV_CAP / 10.0);

    std::vector<std::vector<secondary::ConsensusResult>> results(batch_size);

    for (size_t sample_id = 0; sample_id < batch_size; ++sample_id) {
        std::vector<secondary::ConsensusResult>& hap_results = results[sample_id];
        hap_results.resize(num_haplotypes);

        // Initialize the sequences and qualities for each haplotype.
        for (size_t hap = 0; hap < num_haplotypes; ++hap) {
            secondary::ConsensusResult& result = hap_results[hap];
            result.seq.resize(sequence_length, '*');
            result.quals.resize(sequence_length, '!');
        }

        for (size_t pos = 0; pos < sequence_length; ++pos) {
            for (size_t hap = 0; hap < num_haplotypes; ++hap) {
                secondary::ConsensusResult& result = hap_results[hap];

                const size_t offset = sample_id * sequence_length * num_haplotypes * num_classes +
                                      pos * num_haplotypes * num_classes + hap * num_classes;
                const auto max_it = std::max_element(logits.data() + offset,
                                                     logits.data() + offset + num_classes);

                // Update the base.
                {
                    const int64_t class_index =
                            static_cast<int64_t>(std::distance(logits.data() + offset, max_it));
                    result.seq[pos] = symbols.at(class_index);
                }

                // Update the quality.
                {
                    const double max_prob = static_cast<double>(*max_it);
                    const double err = std::clamp(1.0 - max_prob, min_err, 1.0);
                    const int32_t phred_score =
                            static_cast<int32_t>(std::clamp(-10.0 * std::log10(err), 0.0, QV_CAP)) +
                            33;
                    result.quals[pos] = static_cast<char>(phred_score);
                }
            }
        }
    }

    return results;
}

std::vector<std::vector<secondary::ConsensusResult>> decode_batch_bases_impl(
        const LabelSchemeType label_scheme_type,
        const at::Tensor& logits) {
    if (!logits.defined()) {
        throw std::runtime_error("The input logits tensor is not defined in decode_bases_impl!");
    }

    size_t batch_size = 0;
    size_t sequence_length = 0;
    size_t num_classes = 0;
    size_t num_haplotypes = 0;

    if (logits.sizes().size() == 3) {
        // Input is a tensor of haploid data (legacy), i.e. [B x N x C].
        batch_size = logits.size(0);
        sequence_length = logits.size(1);
        num_haplotypes = 1;
        num_classes = logits.size(2);
    } else if (logits.sizes().size() == 4) {
        // Input is a tensor of haploid/diploid/polyploid data, i.e. [B x N x H x C].
        batch_size = logits.size(0);
        sequence_length = logits.size(1);
        num_haplotypes = logits.size(2);
        num_classes = logits.size(3);
    } else {
        throw std::runtime_error(
                "Wrong input tensor shape! Expected shape: [batch_size x seq_length x "
                "num_classes] for haploid or [batch_size x seq_length x num_haplotypes x "
                "num_classes] for polyploid. Provided shape: " +
                utils::tensor_shape_as_string(logits));
    }

    const std::string symbols = label_scheme_symbols(label_scheme_type);

    const dorado::Span<const float> data(
            logits.data_ptr<float>(), batch_size * sequence_length * num_haplotypes * num_classes);

    return decode_batch_bases_impl(symbols, data, batch_size, sequence_length, num_haplotypes,
                                   num_classes);
}

std::string label_scheme_symbols(const LabelSchemeType label_scheme_type) {
    if (label_scheme_type == LabelSchemeType::HAPLOID) {
        return "*ACGT";
    } else {
        throw std::runtime_error("Unsupported label scheme type!");
    }
}

}  // namespace dorado::secondary
