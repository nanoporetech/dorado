#pragma once

#include "secondary/consensus/consensus_result.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <string>
#include <vector>

namespace dorado::secondary {

enum class LabelSchemeType {
    HAPLOID,
};

class DecoderBase {
public:
    DecoderBase(const LabelSchemeType label_scheme_type);

    std::vector<secondary::ConsensusResult> decode_bases(const at::Tensor& logits) const;

    std::string get_label_scheme_symbols() const;

private:
    LabelSchemeType m_label_scheme_type;
};

LabelSchemeType parse_label_scheme_type(const std::string& type);

std::vector<secondary::ConsensusResult> decode_bases_impl(const LabelSchemeType label_scheme_type,
                                                          const at::Tensor& logits);

std::string label_scheme_symbols(const LabelSchemeType label_scheme_type);

}  // namespace dorado::secondary
