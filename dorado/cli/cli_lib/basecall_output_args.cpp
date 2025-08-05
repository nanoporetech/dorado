#include "basecall_output_args.h"

#include <ctime>
#include <optional>

namespace dorado::cli {

namespace {

constexpr std::string_view OUTPUT_DIR_ARG{"--output-dir"};
constexpr std::string_view EMIT_FASTQ_ARG{"--emit-fastq"};
constexpr std::string_view EMIT_SAM_ARG{"--emit-sam"};
constexpr std::string_view EMIT_SUMMARY_ARG{"--emit-summary"};

void add_emit_types(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument(EMIT_FASTQ_ARG).help("Output in fastq format.").flag();
    parser.visible.add_argument(EMIT_SAM_ARG).help("Output in SAM format.").flag();
}

void add_emit_summary(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument(EMIT_SUMMARY_ARG)
            .help("If specified, a summary file containing the details of the primary "
                  "alignments for each read will be emitted to the root of the output folder.")
            .flag();
}

}  // namespace

void add_basecaller_output_arguments(utils::arg_parse::ArgParser& parser) {
    add_emit_types(parser);
    parser.visible.add_argument("-o", OUTPUT_DIR_ARG)
            .help("Optional output folder which becomes the root of the nested output folder "
                  "structure.");
}

void add_demux_output_arguments(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument("-o", OUTPUT_DIR_ARG)
            .help("Output folder which is the root of the nested output folder structure.")
            .required();
    add_emit_types(parser);
    add_emit_summary(parser);
}

bool get_emit_fastq(const utils::arg_parse::ArgParser& parser) {
    return parser.visible.get<bool>(EMIT_FASTQ_ARG);
}

bool get_emit_sam(const utils::arg_parse::ArgParser& parser) {
    return parser.visible.get<bool>(EMIT_SAM_ARG);
}

bool get_emit_summary(const utils::arg_parse::ArgParser& parser) {
    return parser.visible.get<bool>(EMIT_SUMMARY_ARG);
}

std::optional<std::string> get_output_dir(const utils::arg_parse::ArgParser& parser) {
    return parser.visible.present<std::string>(OUTPUT_DIR_ARG);
}

}  // namespace dorado::cli