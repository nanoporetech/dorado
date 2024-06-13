#pragma once

#include "utils/arg_parse_ext.h"
#include "utils/hts_file.h"

#include <memory>

namespace dorado::cli {

void add_basecaller_output_arguments(utils::arg_parse::ArgParser& parser);

std::unique_ptr<utils::HtsFile> extract_hts_file(const utils::arg_parse::ArgParser& parser);

}  // namespace dorado::cli