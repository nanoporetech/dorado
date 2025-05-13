#pragma once

#include "secondary/consensus/consensus_result.h"
#include "utils/span.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <string>
#include <vector>

namespace dorado::secondary {

enum class LabelSchemeType {
    HAPLOID,
    DIPLOID,
};

class DecoderBase {
public:
    DecoderBase(const LabelSchemeType label_scheme_type);

    /**
     * \brief Given an input batch tensor of probabilities this function decodes the bases and qualities
     *          for each sample in the batch and each haplotype.
     * \param logits Input probabilities, shape of the tensor can be either [B x N x C] for a single
     *                  haplotype (legacy) or [B x N x H x C] for 1 or more haplotypes.
     *                  Here:
     *                      - B - batch size (number of samples, can be 1).
     *                      - N - number of positions for each sample
     *                      - H - number of haplotypes (1 for haploid)
     *                      - C - number of classes (e.g. 5 for `*ACGT`)
     * \returns Outer vector contains data for each sample in the input batch (B) and the inner vector contains
     *          one element for each haplotype (H). For an input tensor of shape [B x N x C] the value of H = 1.
     */
    std::vector<std::vector<secondary::ConsensusResult>> decode_bases(
            const at::Tensor& logits) const;

    std::string get_label_scheme_symbols() const;

private:
    LabelSchemeType m_label_scheme_type;
};

LabelSchemeType parse_label_scheme_type(const std::string& type);

/**
 * \brief Decodes consensus bases for an input tensor of forms [B x N x C] or [B x N x H x C].
 *          Here:
 *              - B - batch size (number of samples)
 *              - N - number of positions for each sample
 *              - H - number of haplotypes (1 for haploid)
 *              - C - number of classes (e.g. 5 for `*ACGT`)
 *          This function works for non-batch data (e.g. number of samples is 1) and for
 *          haploid/diploid/polyploid use cases as well.
 */
std::vector<std::vector<secondary::ConsensusResult>> decode_batch_bases_impl(
        const LabelSchemeType label_scheme_type,
        const at::Tensor& logits);

/**
 * \brief Decodes consensus bases for an input tensor of form [B x N x H x C], where the tensor is
 *          given as raw data (span of a raw pointer).
 *          Here:
 *              - B - batch size (number of samples)
 *              - N - number of positions for each sample
 *              - H - number of haplotypes (1 for haploid)
 *              - C - number of classes (e.g. 5 for `*ACGT`)
 *          This function works for non-batch data (e.g. number of samples is 1) and for
 *          haploid/diploid/polyploid use cases as well.
 */
std::vector<std::vector<secondary::ConsensusResult>> decode_batch_bases_impl(
        const std::string& symbols,
        const dorado::Span<const float> logits,
        const size_t batch_size,
        const size_t sequence_length,
        const size_t num_haplotypes,
        const size_t num_classes);

std::string label_scheme_symbols(const LabelSchemeType label_scheme_type);

}  // namespace dorado::secondary
