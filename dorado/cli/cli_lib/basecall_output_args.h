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

}  // namespace dorado::cli