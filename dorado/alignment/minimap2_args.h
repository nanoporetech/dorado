#pragma once

#include "Minimap2Options.h"
#include "utils/arg_parse_ext.h"

#include <optional>
#include <string>
#include <vector>

using namespace std::literals;

namespace dorado::alignment::mm2 {

// Retrieves the minimap2 options string from the given args and assigns the remaining args to the given out param.
std::string extract_options_string_arg(const std::vector<std::string>& args,
                                       std::vector<std::string>& remaining_args);

// Adds the minimap2 options string argument to the parser.
void add_options_string_arg(utils::arg_parse::ArgParser& parser);

Minimap2Options parse_options(const std::string& minimap2_option_string);

std::optional<Minimap2Options> try_parse_options(const std::string& minimap2_option_string,
                                                 std::string& error_message);

// Returns the help text associated with the supported minimap options
std::string get_help_message();

// Helper functions for ErrorCorrectionMapperNode which uses options we don't yet support in the command line
void apply_cs_option(Minimap2Options& options, const std::string& cs_opt);
void apply_dual_option(Minimap2Options& options, const std::string& dual_yes_or_no);

// Returns true in the minimap global flag 'MM_DBG_PRINT_ALN_SEQ' is set
bool print_aln_seq();

}  // namespace dorado::alignment::mm2