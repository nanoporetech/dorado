#include "basecall_output_args.h"

#include <argparse/argparse.hpp>

#include <ctime>
#include <optional>

namespace dorado::cli {

namespace {

constexpr std::string_view OUTPUT_DIR_ARG{"--output-dir"};
constexpr std::string_view EMIT_FASTQ_ARG{"--emit-fastq"};
constexpr std::string_view EMIT_SAM_ARG{"--emit-sam"};
constexpr std::string_view EMIT_SUMMARY_ARG{"--emit-summary"};

void add_fastq_emit_type(argparse::ArgumentParser& parser) {
    parser.add_argument(EMIT_FASTQ_ARG).help("Output in fastq format.").flag();
}

void add_sam_emit_type(argparse::ArgumentParser& parser) {
    parser.add_argument(EMIT_SAM_ARG).help("Output in SAM format.").flag();
}

const std::string emit_summary_help_text =
        "If specified, a summary file containing the details of the primary "
        "alignments for each read will be emitted to the root of the --output-dir folder.";

argparse::Argument& add_emit_summary(argparse::ArgumentParser& parser) {
    return parser.add_argument(EMIT_SUMMARY_ARG).help(emit_summary_help_text).flag();
}

const std::string out_dir_help_text =
        "Output folder which becomes the root of the nested output folder structure.";

argparse::Argument& add_output_dir_argument(argparse::ArgumentParser& parser) {
    return parser.add_argument("-o", OUTPUT_DIR_ARG).help(out_dir_help_text);
}

}  // namespace

void add_basecaller_output_arguments(argparse::ArgumentParser& parser, bool allow_summary) {
    add_fastq_emit_type(parser);
    add_sam_emit_type(parser);
    if (allow_summary) {
        auto& summary_arg = add_emit_summary(parser);
        summary_arg.help(emit_summary_help_text +
                         " If --output-dir is not set, the summary file is placed in the current "
                         "working directory.");
    }
    add_output_dir_argument(parser);
}

void add_demux_output_arguments(argparse::ArgumentParser& parser) {
    add_fastq_emit_type(parser);
    add_sam_emit_type(parser);
    add_emit_summary(parser);

    auto& out_dir_arg = add_output_dir_argument(parser);
    out_dir_arg.required();
}

void add_aligner_output_arguments(argparse::ArgumentParser& parser) {
    add_sam_emit_type(parser);
    add_emit_summary(parser);

    auto& out_dir_arg = add_output_dir_argument(parser);
    out_dir_arg.help(out_dir_help_text +
                     " Required if the 'reads' positional argument is a folder.");
}

std::optional<std::string> get_output_dir(const argparse::ArgumentParser& parser) {
    return parser.present<std::string>(OUTPUT_DIR_ARG);
}

bool get_emit_fastq(const argparse::ArgumentParser& parser) {
    return parser.get<bool>(EMIT_FASTQ_ARG);
}

bool get_emit_sam(const argparse::ArgumentParser& parser) { return parser.get<bool>(EMIT_SAM_ARG); }

bool get_emit_summary(const argparse::ArgumentParser& parser) {
    return parser.get<bool>(EMIT_SUMMARY_ARG);
}

}  // namespace dorado::cli