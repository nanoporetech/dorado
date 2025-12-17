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
constexpr std::string_view EMIT_MOVES_ARG{"--emit-moves"};

void add_sam_emit_type(argparse::ArgumentParser& parser) {
    parser.add_argument(EMIT_SAM_ARG).help("Output in SAM format.").flag();
}

void add_emit_hts_types(argparse::ArgumentParser& parser) {
    auto& emit_mutex = parser.add_mutually_exclusive_group();
    emit_mutex.add_argument(EMIT_FASTQ_ARG).help("Output in FASTQ format.").flag();
    emit_mutex.add_argument(EMIT_SAM_ARG).help("Output in SAM format.").flag();
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

// Attempt to get a bool flag argument returning defalt_value if the argument hasn't been added.
bool get_optional_flag(const argparse::ArgumentParser& parser,
                       std::string_view arg,
                       bool default_value) {
    try {
        return parser.get<bool>(arg);
    } catch (const std::logic_error&) {
        return default_value;
    }
}

}  // namespace

void add_basecaller_output_arguments(argparse::ArgumentParser& parser) {
    add_emit_hts_types(parser);
    parser.add_argument(EMIT_MOVES_ARG).help("Write the move table to the 'mv' tag.").flag();

    auto& summary_arg = add_emit_summary(parser);
    summary_arg.help(emit_summary_help_text +
                     " If --output-dir is not set, the summary file is placed in the current "
                     "working directory.");

    add_output_dir_argument(parser);
}

void add_duplex_output_arguments(argparse::ArgumentParser& parser) {
    add_emit_hts_types(parser);
    add_output_dir_argument(parser);
}

void add_demux_output_arguments(argparse::ArgumentParser& parser) {
    add_emit_hts_types(parser);
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

EmitArgs get_emit_args(const argparse::ArgumentParser& parser) {
    EmitArgs emit;

    emit.sam = get_optional_flag(parser, EMIT_SAM_ARG, false);
    emit.fastq = get_optional_flag(parser, EMIT_FASTQ_ARG, false);
    emit.bam = !(emit.sam || emit.fastq);

    emit.moves = get_optional_flag(parser, EMIT_MOVES_ARG, false);
    emit.summary = get_optional_flag(parser, EMIT_SUMMARY_ARG, false);
    return emit;
};

std::optional<std::string> get_output_dir(const argparse::ArgumentParser& parser) {
    return parser.present<std::string>(OUTPUT_DIR_ARG);
}

}  // namespace dorado::cli