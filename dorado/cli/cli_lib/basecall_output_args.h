#pragma once

#include <optional>
#include <string>

namespace argparse {
class ArgumentParser;
}

namespace dorado::cli {

void add_output_dir_argument(argparse::ArgumentParser& parser);
std::optional<std::string> get_output_dir(const argparse::ArgumentParser& parser);

void add_basecaller_output_arguments(argparse::ArgumentParser& parser);
void add_demux_output_arguments(argparse::ArgumentParser& parser);

bool get_emit_fastq(const argparse::ArgumentParser& parser);
bool get_emit_sam(const argparse::ArgumentParser& parser);
bool get_emit_summary(const argparse::ArgumentParser& parser);

}  // namespace dorado::cli