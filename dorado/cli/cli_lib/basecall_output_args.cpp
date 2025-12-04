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

void add_emit_summary(argparse::ArgumentParser& parser, bool requires_output_dir) {
    std::string help_text(
            "If specified, a summary file containing the details of the primary "
            "alignments for each read will be emitted to the root of the --output-dir folder");
    if (requires_output_dir) {
        help_text.append(".");
    } else {
        help_text.append(" (or to the current working directory if --output-dir is not set).");
    }
    parser.add_argument(EMIT_SUMMARY_ARG).help(help_text).flag();
}

void add_output_dir_argument(argparse::ArgumentParser& parser, bool required) {
    auto& arg = parser.add_argument("-o", OUTPUT_DIR_ARG);
    std::string help_text =
            "Output folder which becomes the root of the nested output folder structure.";
    if (required) {
        arg.required();
    } else {
        help_text.append(" (Optional)");
    }
    arg.help(help_text);
}

}  // namespace

void add_basecaller_output_arguments(argparse::ArgumentParser& parser, bool allow_summary) {
    add_fastq_emit_type(parser);
    add_sam_emit_type(parser);
    if (allow_summary) {
        add_emit_summary(parser, false);
    }
    add_output_dir_argument(parser, false);
}

void add_demux_output_arguments(argparse::ArgumentParser& parser) {
    add_fastq_emit_type(parser);
    add_sam_emit_type(parser);
    add_emit_summary(parser, true);
    add_output_dir_argument(parser, true);
}

void add_aligner_output_arguments(argparse::ArgumentParser& parser) {
    add_sam_emit_type(parser);
    add_emit_summary(parser, true);
    add_output_dir_argument(parser, false);
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