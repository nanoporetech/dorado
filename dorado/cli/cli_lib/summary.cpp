#include "summary/summary.h"

#include "cli/cli.h"
#include "dorado_version.h"
#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/SummaryFileWriter.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/nodes/WriterNode.h"
#include "utils/log_utils.h"
#include "utils/time_utils.h"
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
    const auto file = dorado::HtsFilePtr(hts_open(path.c_str(), "r"));
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

    auto command_line_cl =
            utils::extract_pg_keys_from_hdr(reader.header(), {"CL"}, "ID", "basecaller");
    // If dorado was run with --min-qscore option, parse the value so we can re-evaluate the pass/fail criterion
    int minimum_qscore = 0;
    std::stringstream cl{command_line_cl["CL"]};
    std::string out;
    while (cl.good()) {
        cl >> std::quoted(out);
        if (out == "--min-qscore") {
            cl >> std::quoted(out);
            minimum_qscore = std::atoi(out.c_str());
            break;
        }
    }

    auto read_groups = utils::parse_read_groups(reader.header());

    auto update_read_attributes = [groups = std::move(read_groups), minimum_qscore](HtsData &data) {
        if (const auto rg_tag = bam_aux_get(data.bam_ptr.get(), "RG"); rg_tag != nullptr) {
            const std::string rg_tag_value = bam_aux2Z(rg_tag);
            const auto &read_group = groups.at(rg_tag_value);
            data.read_attrs.protocol_run_id = read_group.run_id;
            data.read_attrs.flowcell_id = read_group.flowcell_id;
            data.read_attrs.experiment_id = read_group.experiment_id;
            data.read_attrs.sample_id = read_group.sample_id;
            data.read_attrs.position_id = read_group.position_id;
            data.read_attrs.model_stride = read_group.model_stride;

            if (const auto qs_tag = bam_aux_get(data.bam_ptr.get(), "qs"); qs_tag != nullptr) {
                const float qscore = static_cast<float>(bam_aux2f(qs_tag));
                data.read_attrs.is_status_pass = qscore >= minimum_qscore;
            }

            try {
                if (const auto st_tag = bam_aux_get(data.bam_ptr.get(), "st"); st_tag != nullptr) {
                    const std::string read_start_time_str = bam_aux2Z(st_tag);
                    const auto acq_start_time = utils::get_unix_time_ms_from_string_timestamp(
                            read_group.acq_start_time);
                    const auto read_start_time =
                            utils::get_unix_time_ms_from_string_timestamp(read_start_time_str);
                    data.read_attrs.start_time_ms = read_start_time - acq_start_time;
                }
            } catch (...) {
                // can't parse something, ignore start_time and continue
            }
        }
    };

    auto update_barcoding_fields = [hdr = reader.header()](HtsData &data) {
        if (const auto rg_tag = bam_aux_get(data.bam_ptr.get(), "RG"); rg_tag != nullptr) {
            const std::string rg_tag_value = bam_aux2Z(rg_tag);
            KString ks_wrapper(100000);
            auto &ks = ks_wrapper.get();

            data.barcoding_result = std::make_shared<BarcodeScoreResult>();
            if (sam_hdr_find_tag_id(hdr, "RG", "ID", rg_tag_value.c_str(), "SM", &ks) == 0) {
                data.barcoding_result->barcode_name = std::string(ks.s, ks.l);
            }
            if (sam_hdr_find_tag_id(hdr, "RG", "ID", rg_tag_value.c_str(), "al", &ks) == 0) {
                data.barcoding_result->alias = std::string(ks.s, ks.l);
            }
            if (sam_hdr_find_tag_id(hdr, "RG", "ID", rg_tag_value.c_str(), "bk", &ks) == 0) {
                data.barcoding_result->kit = std::string(ks.s, ks.l);
            }
        }
    };

    auto update_alignment_fields = [counts_by_read = std::move(alignment_counts)](HtsData &data) {
        if (counts_by_read.has_value()) {
            const auto alignment_counts_it =
                    counts_by_read->find(bam_get_qname(data.bam_ptr.get()));
            if (alignment_counts_it != std::end(*counts_by_read)) {
                const auto &counts = alignment_counts_it->second;
                data.read_attrs.num_alignments = counts[0];
                data.read_attrs.num_secondary_alignments = counts[1];
                data.read_attrs.num_supplementary_alignments = counts[2];
            }
        }
    };

    reader.add_read_initialiser(std::move(update_read_attributes));

    std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
    {
        using namespace hts_writer;
        SummaryFileWriter::FieldFlags flags =
                SummaryFileWriter::BASECALLING_FIELDS | SummaryFileWriter::EXPERIMENT_FIELDS;
        if (reader.is_aligned) {
            flags |= SummaryFileWriter::ALIGNMENT_FIELDS;
            reader.add_read_initialiser(std::move(update_alignment_fields));
        }

        // If dorado was run with --estimate-poly-a option, output polyA related fields in the summary
        if (command_line_cl["CL"].find("estimate-poly-a") != std::string::npos) {
            flags |= SummaryFileWriter::POLYA_FIELDS;
        }

        SamHdrSharedPtr shared_hdr(sam_hdr_dup(reader.header()));
        auto hdr = const_cast<sam_hdr_t *>(shared_hdr.get());
        int num_rg_lines = sam_hdr_count_lines(hdr, "RG");
        KString tag_wrapper(100000);
        auto &tag_value = tag_wrapper.get();
        for (int i = 0; i < num_rg_lines; ++i) {
            if (sam_hdr_find_tag_pos(hdr, "RG", i, "SM", &tag_value) == 0) {
                flags |= SummaryFileWriter::BARCODING_FIELDS;
                reader.add_read_initialiser(std::move(update_barcoding_fields));
                break;
            }
        }

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
