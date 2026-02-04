#pragma once

#include <optional>
#include <string>

namespace argparse {
class ArgumentParser;
}

namespace dorado::cli {

struct EmitArgs {
    bool bam{false};
    bool sam{false};
    bool cram{false};
    bool fastq{false};
    bool moves{false};
    bool summary{false};
};

EmitArgs get_emit_args(const argparse::ArgumentParser& parser);

void add_basecaller_output_arguments(argparse::ArgumentParser& parser);
void add_duplex_output_arguments(argparse::ArgumentParser& parser);
void add_demux_output_arguments(argparse::ArgumentParser& parser);
void add_aligner_output_arguments(argparse::ArgumentParser& parser);

std::optional<std::string> get_output_dir(const argparse::ArgumentParser& parser);

// True if `--emit-cram` is set and the ref path ends in `.mmi`
bool emit_cram_with_mmi_reference(const EmitArgs& emit, const std::optional<std::string>& ref);

}  // namespace dorado::cli