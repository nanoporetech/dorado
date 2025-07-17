#pragma once

#include "utils/arg_parse_ext.h"

namespace dorado::cli {

void add_basecaller_output_arguments(utils::arg_parse::ArgParser& parser);

bool get_emit_fastq(const utils::arg_parse::ArgParser& parser);
bool get_emit_sam(const utils::arg_parse::ArgParser& parser);
std::optional<std::string> get_output_dir(const utils::arg_parse::ArgParser& parser);

}  // namespace dorado::cli