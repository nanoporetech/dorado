#include "summary/summary.h"

#include "cli/cli.h"
#include "dorado_version.h"
#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_writer/SummaryFileWriter.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/nodes/WriterNode.h"
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

using AlignmentCounts = hts_writer::SummaryFileWriter::AlignmentCounts;
namespace {
std::optional<AlignmentCounts> get_alignment_counts(const std::string &path) {
    auto file = dorado::HtsFilePtr(hts_open(path.c_str(), "r"));
    if (file->format.format != htsExactFormat::sam && file->format.format != htsExactFormat::bam) {
        return std::nullopt;
    }

    dorado::SamHdrPtr header(sam_hdr_read(file.get()));
    if (header->n_targets == 0) {
        return std::nullopt;
    }

    AlignmentCounts alignment_counts;
    BamPtr record(bam_init1());
    while (sam_read1(file.get(), header.get(), record.get()) >= 0) {
        if (record->core.flag & BAM_FUNMAP) {
            continue;
        }
        auto &read_counts = alignment_counts[bam_get_qname(record.get())];
        if (record->core.flag & BAM_FSUPPLEMENTARY) {
            ++read_counts[2];
        }
        if (record->core.flag & BAM_FSECONDARY) {
            ++read_counts[1];
        }
        ++read_counts[0];
    }
    if (alignment_counts.empty()) {
        return std::nullopt;
    }

    return alignment_counts;
}
}  // namespace

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

    std::optional<AlignmentCounts> alignment_counts;
    if (!reads.empty()) {
        if (!std::filesystem::exists(reads)) {
            spdlog::error("Unable to open file '{}', no such file.", reads);
            return EXIT_FAILURE;
        }

        if (std::filesystem::is_directory(reads)) {
            spdlog::error("Failed to open file '{}', found a directory instead.", reads);
            return EXIT_FAILURE;
        }
        alignment_counts = get_alignment_counts(reads);
    } else if (utils::is_fd_tty(stdin)) {
        // Only allow `reads` to be empty if we're accepting input from a pipe
        std::cout << parser << '\n';
        return EXIT_FAILURE;
    } else {
        reads = "-";
    }

    HtsReader reader(reads, std::nullopt);
    std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
    {
        using namespace hts_writer;
        SummaryFileWriter::FieldFlags flags =
                SummaryFileWriter::BASECALLING_FIELDS | SummaryFileWriter::EXPERIMENT_FIELDS;
        if (reader.is_aligned) {
            flags |= SummaryFileWriter::ALIGNMENT_FIELDS;
        }

        auto command_line_cl =
                utils::extract_pg_keys_from_hdr(reader.header(), {"CL"}, "ID", "basecaller");

        // If dorado was run with --estimate-poly-a option, output polyA related fields in the summary
        if (command_line_cl["CL"].find("estimate-poly-a") != std::string::npos) {
            flags |= SummaryFileWriter::POLYA_FIELDS;
        }

        auto hdr = sam_hdr_dup(reader.header());
        int num_rg_lines = sam_hdr_count_lines(hdr, "RG");
        KString tag_wrapper(100000);
        auto &tag_value = tag_wrapper.get();
        for (int i = 0; i < num_rg_lines; ++i) {
            if (sam_hdr_find_tag_pos(hdr, "RG", i, "SM", &tag_value) == 0) {
                flags |= SummaryFileWriter::BARCODING_FIELDS;
                break;
            }
        }

        SamHdrSharedPtr shared_hdr(hdr);
        auto summary_writer =
                std::make_unique<hts_writer::SummaryFileWriter>(std::cout, flags, alignment_counts);
        summary_writer->set_header(shared_hdr);
        writers.push_back(std::move(summary_writer));
    }

    PipelineDescriptor pipeline_desc;
    pipeline_desc.add_node<WriterNode>({}, std::move(writers));

    auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    reader.read(*pipeline, 0, false, nullptr, true);
    pipeline->terminate({.fast = utils::AsyncQueueTerminateFast::No});

    return EXIT_SUCCESS;
}

}  // namespace dorado
