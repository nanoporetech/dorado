#pragma once

#include "polish/consensus_result.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <string>
#include <vector>

namespace dorado::polisher {

enum class LabelSchemeType {
    HAPLOID,
};

class DecoderBase {
public:
    DecoderBase(const LabelSchemeType label_scheme_type);

    std::vector<ConsensusResult> decode_bases(const torch::Tensor& logits) const;

private:
    LabelSchemeType m_label_scheme_type;
};

LabelSchemeType parse_label_scheme_type(const std::string& type);

std::vector<ConsensusResult> decode_bases_impl(const LabelSchemeType label_scheme_type,
                                               const torch::Tensor& logits);

}  // namespace dorado::polisher