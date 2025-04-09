#pragma once

#include "consensus_result.h"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace dorado::secondary {

std::string extract_draft_with_gaps(const std::string& draft,
                                    const std::vector<int64_t>& positions_major,
                                    const std::vector<int64_t>& positions_minor);

std::vector<bool> variant_columns(const std::vector<int64_t>& minor,
                                  const std::string_view reference,
                                  const std::string_view prediction);

std::vector<bool> find_polyploid_variants(
        const std::vector<int64_t>& positions_minor,
        const std::string_view ref_seq_with_gaps,
        const std::vector<std::string_view>& cons_seqs_with_gaps,
        const std::optional<std::unordered_set<char>>& allowed_symbol_set);

std::vector<bool> find_polyploid_variants(
        const std::vector<int64_t>& positions_minor,
        const std::string_view ref_seq_with_gaps,
        const std::vector<ConsensusResult>& cons_seqs_with_gaps,
        const std::optional<std::unordered_set<char>>& allowed_symbol_set);

void print_slice(std::ostream& os,
                 const std::string_view ref_seq_with_gaps,
                 const std::vector<std::string_view>& cons_seqs,
                 const std::vector<int64_t>& pos_major,
                 const std::vector<int64_t>& pos_minor,
                 const std::vector<bool>& is_var,
                 int64_t slice_start,
                 int64_t slice_end,
                 int64_t rstart,
                 int64_t rend);

}  // namespace dorado::secondary
