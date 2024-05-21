#pragma once

#include "Minimap2Options.h"
#include "utils/arg_parse_ext.h"

#include <string>
#include <string_view>
#include <vector>

using namespace std::literals;

namespace dorado::alignment {

constexpr inline std::string_view MM2_OPTS_ARG = "--mm2-opts";

std::string extract_minimap2_options_string_arg(const std::vector<std::string>& args,
                                                std::vector<std::string>& remaining_args);

void add_minimap2_options_string_arg(utils::arg_parse::ArgParser& parser);

Minimap2Options process_minimap2_options_string(const std::string& minimap2_option_string);

// Helper functions for ErrorCorrectionMapperNode which uses options we don't yet support in the command line
void apply_minimap2_cs_option(Minimap2Options& options, const std::string& cs_opt);
void apply_minimap2_dual_option(Minimap2Options& options, const std::string& dual_yes_or_no);

bool minimap2_print_aln_seq();

}  // namespace dorado::alignment