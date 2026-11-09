#pragma once

#include "bam_info.h"

#include <iosfwd>
#include <string>
#include <unordered_set>

namespace dorado::secondary {

void print_basecaller_models(std::ostream& os,
                             const std::unordered_set<std::string>& basecaller_models,
                             const std::string& delimiter);

void throw_unless_single_model_present(const BamInfo& bam_info);

void check_stereo_models(const BamInfo& bam_inf, bool throw_error);

}  // namespace dorado::secondary
