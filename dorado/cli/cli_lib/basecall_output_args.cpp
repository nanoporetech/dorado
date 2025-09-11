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

void add_emit_types(argparse::ArgumentParser& parser) {
    parser.add_argument(EMIT_FASTQ_ARG).help("Output in fastq format.").flag();
    parser.add_argument(EMIT_SAM_ARG).help("Output in SAM format.").flag();
}

void add_emit_summary(argparse::ArgumentParser& parser) {
    parser.add_argument(EMIT_SUMMARY_ARG)
            .help("If specified, a summary file containing the details of the primary "
                  "alignments for each read will be emitted to the root of the output folder.")
            .flag();
}

}  // namespace

void add_output_dir_argument(argparse::ArgumentParser& parser) {
    parser.add_argument("-o", OUTPUT_DIR_ARG)
            .help("Optional output folder which becomes the root of the nested output folder "
                  "structure.");
}

std::optional<std::string> get_output_dir(const argparse::ArgumentParser& parser) {
    return parser.present<std::string>(OUTPUT_DIR_ARG);
}

void add_basecaller_output_arguments(argparse::ArgumentParser& parser) {
    add_emit_types(parser);
    add_output_dir_argument(parser);
}

void add_demux_output_arguments(argparse::ArgumentParser& parser) {
    add_emit_types(parser);
    add_emit_summary(parser);
    add_output_dir_argument(parser);
}

bool get_emit_fastq(const argparse::ArgumentParser& parser) {
    return parser.get<bool>(EMIT_FASTQ_ARG);
}

bool get_emit_sam(const argparse::ArgumentParser& parser) { return parser.get<bool>(EMIT_SAM_ARG); }

bool get_emit_summary(const argparse::ArgumentParser& parser) {
    return parser.get<bool>(EMIT_SUMMARY_ARG);
}

}  // namespace dorado::cli