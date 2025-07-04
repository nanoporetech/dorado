#include "basecall_output_args.h"

#include <ctime>
#include <optional>

namespace dorado::cli {

namespace {

constexpr std::string_view OUTPUT_DIR_ARG{"--output-dir"};
constexpr std::string_view EMIT_FASTQ_ARG{"--emit-fastq"};
constexpr std::string_view EMIT_SAM_ARG{"--emit-sam"};

}  // namespace

void add_basecaller_output_arguments(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument(EMIT_FASTQ_ARG)
            .help("Output in fastq format.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument(EMIT_SAM_ARG)
            .help("Output in SAM format.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("-o", OUTPUT_DIR_ARG)
            .help("Optional output folder, if specified output will be written to a calls file "
                  "(calls_<timestamp>.sam|.bam|.fastq) in the given folder.");
}

bool get_emit_fastq(const utils::arg_parse::ArgParser& parser) {
    return parser.visible.get<bool>(EMIT_FASTQ_ARG);
}

bool get_emit_sam(const utils::arg_parse::ArgParser& parser) {
    return parser.visible.get<bool>(EMIT_SAM_ARG);
}

std::optional<std::string> get_output_dir(const utils::arg_parse::ArgParser& parser) {
    return parser.visible.present<std::string>(OUTPUT_DIR_ARG);
}

}  // namespace dorado::cli