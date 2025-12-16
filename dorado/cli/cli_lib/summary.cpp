#include "cli/cli.h"
#include "cli/utils/cli_utils.h"
#include "dorado_version.h"
#include "hts_utils/HeaderMapper.h"
#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/SummaryFileWriter.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadInitialiser.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/nodes/WriterNode.h"
#include "summary_info.h"
#include "utils/log_utils.h"
#include "utils/tty_utils.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <cctype>
#include <csignal>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace dorado {

volatile sig_atomic_t interrupt = 0;

int summary(int argc, char *argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("reads")
            .help("SAM/BAM file produced by dorado basecaller.")
            .nargs(argparse::nargs_pattern::optional)
            .default_value(std::string{});
    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .flag()
            .action([&](const auto &) { ++verbosity; })
            .append();
    parser.add_argument("-r", "--recursive")
            .help("If the 'reads' positional argument is a folder any subfolders will also be "
                  "searched for input files.")
            .flag();
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto reads(parser.get<std::string>("reads"));

    // Only allow `reads` to be empty if we're accepting input from a pipe
    if (reads.empty() && utils::is_fd_tty(stdin)) {
        std::cout << parser << '\n';
        return EXIT_FAILURE;
    }

    const auto all_files = cli::collect_inputs(reads, parser.get<bool>("recursive"));
    if (all_files.empty()) {
        spdlog::info("No input files found");
        return EXIT_SUCCESS;
    }

    auto [flags, alignment_counts] = cli::make_summary_info(all_files);
    std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
    {
        auto summary_writer = std::make_unique<hts_writer::SummaryFileWriter>(std::cout, flags);
        writers.push_back(std::move(summary_writer));
    }

    PipelineDescriptor pipeline_desc;
    auto writer_node = pipeline_desc.add_node<WriterNode>({}, std::move(writers));

    auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    if (!reads.empty()) {
        utils::HeaderMapper header_mapper(all_files, false);
        pipeline->get_node_ref<WriterNode>(writer_node)
                .set_dynamic_header(header_mapper.get_merged_headers_map());
    } else {
        spdlog::warn(
                "Reading from stdin: unable to check for polyA, barcode or alignment information. "
                "Some columns will be unavailable.");
    }

    using namespace hts_writer;
    for (const auto &input_file : all_files) {
        HtsReader reader(input_file.string(), std::nullopt);
        ReadInitialiser read_initialiser(reader.header(), alignment_counts);
        reader.add_read_initialiser([&read_initialiser](HtsData &data) {
            read_initialiser.update_read_attributes(data);
        });

        if (flags & SummaryFileWriter::ALIGNMENT_FIELDS) {
            reader.add_read_initialiser([&read_initialiser](HtsData &data) {
                read_initialiser.update_alignment_fields(data);
            });
        }

        if (flags & SummaryFileWriter::BARCODING_FIELDS) {
            reader.add_read_initialiser([&read_initialiser](HtsData &data) {
                read_initialiser.update_barcoding_fields(data);
            });
        }

        reader.read(*pipeline, 0, false, nullptr, true);
    }
    pipeline->terminate({.fast = utils::AsyncQueueTerminateFast::No});

    return EXIT_SUCCESS;
}

}  // namespace dorado
