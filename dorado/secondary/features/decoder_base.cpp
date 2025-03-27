#include "decoder_base.h"

#include "torch_utils/tensor_utils.h"
#include "utils/span.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace dorado::secondary {

DecoderBase::DecoderBase(const LabelSchemeType label_scheme_type)
        : m_label_scheme_type(label_scheme_type) {}

std::vector<secondary::ConsensusResult> DecoderBase::decode_bases(const at::Tensor& logits) const {
    return decode_bases_impl(m_label_scheme_type, logits);
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

namespace {

std::vector<secondary::ConsensusResult> decode_bases_impl(const LabelSchemeType label_scheme_type,
                                                          const dorado::Span<const float> logits,
                                                          const size_t num_samples,
                                                          const size_t sequence_length,
                                                          const size_t num_classes) {
    const std::string label_scheme = label_scheme_symbols(label_scheme_type);

    constexpr double QV_CAP = 70.0;
    const double min_err = std::pow(10.0, -QV_CAP / 10.0);

    std::vector<secondary::ConsensusResult> results;
    results.reserve(num_samples);
    for (size_t sample_id = 0; sample_id < num_samples; ++sample_id) {
        auto& result = results.emplace_back();
        result.seq.resize(sequence_length, '*');
        result.quals.resize(sequence_length, '!');

        for (size_t j = 0; j < sequence_length; ++j) {
            const size_t offset = sample_id * sequence_length * num_classes + j * num_classes;
            const auto max_it =
                    std::max_element(logits.data() + offset, logits.data() + offset + num_classes);

            // Update the base.
            {
                const int64_t class_index =
                        static_cast<int64_t>(std::distance(logits.data() + offset, max_it));
                result.seq[j] = label_scheme.at(class_index);
            }

            // Update the quality.
            {
                const double max_prob = static_cast<double>(*max_it);
                const double err = std::clamp(1.0 - max_prob, min_err, 1.0);
                const int32_t phred_score =
                        static_cast<int32_t>(std::clamp(-10.0 * std::log10(err), 0.0, QV_CAP)) + 33;
                result.quals[j] = static_cast<char>(phred_score);
            }
        }
    }

    return results;
}

}  // namespace

std::vector<secondary::ConsensusResult> decode_bases_impl(const LabelSchemeType label_scheme_type,
                                                          const at::Tensor& logits) {
    if (logits.sizes().size() != 3) {
        throw std::runtime_error(
                "Wrong input tensor shape! Expected shape: [num_samples x seq_length x "
                "num_classes]. Provided shape: " +
                utils::tensor_shape_as_string(logits));
    }

    const size_t num_samples = logits.size(0);
    const size_t sequence_length = logits.size(1);
    const size_t num_classes = logits.size(2);

    const dorado::Span<const float> data(logits.data_ptr<float>(),
                                         num_samples * sequence_length * num_classes);

    return decode_bases_impl(label_scheme_type, data, num_samples, sequence_length, num_classes);
}

std::string label_scheme_symbols(const LabelSchemeType label_scheme_type) {
    if (label_scheme_type == LabelSchemeType::HAPLOID) {
        return "*ACGT";
    } else {
        throw std::runtime_error("Unsupported label scheme type!");
    }
}

}  // namespace dorado::secondary
