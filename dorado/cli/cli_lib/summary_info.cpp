#include "summary_info.h"

#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "read_pipeline/base/HtsReader.h"
#include "utils/log_utils.h"

namespace dorado::cli {

namespace {

void update_alignment_counts(HtsReader& file, AlignmentCounts& alignment_counts) {
    if (file.exact_format() != htsExactFormat::sam && file.exact_format() != htsExactFormat::bam &&
        file.exact_format() != htsExactFormat::cram) {
        return;
    } else if (!file.is_aligned()) {
        return;
    }

    std::size_t num_reads = 0;
    while (file.read()) {
        if (++num_reads % 50'000 == 0) {
            DORADO_LOG(debug, "Preprocessed {} reads", num_reads);
        }

        const auto& record = file.record;
        if (record->core.flag & BAM_FUNMAP) {
            continue;
        }
        auto& read_counts = alignment_counts[bam_get_qname(record)];
        if (record->core.flag & BAM_FSUPPLEMENTARY) {
            ++read_counts[2];
        }
        if (record->core.flag & BAM_FSECONDARY) {
            ++read_counts[1];
        }
        ++read_counts[0];
    }
}

}  // namespace

std::tuple<hts_writer::SummaryFileWriter::FieldFlags, AlignmentCounts> make_summary_info(
        const std::vector<std::filesystem::path>& all_files) {
    using hts_writer::SummaryFileWriter;

    SummaryFileWriter::FieldFlags flags =
            SummaryFileWriter::BASECALLING_FIELDS | SummaryFileWriter::EXPERIMENT_FIELDS;
    AlignmentCounts alignment_counts;
    if (!(all_files.size() == 1 && all_files[0] == "-")) {
        DORADO_LOG(info, "Preprocessing records...");

        // We're the only thing running, so use all the threads.
        const std::size_t num_threads = std::thread::hardware_concurrency();

        for (const auto& input_file : all_files) {
            HtsReader reader(input_file.string(), std::nullopt, num_threads);
            update_alignment_counts(reader, alignment_counts);

            if (reader.is_aligned()) {
                flags |= SummaryFileWriter::ALIGNMENT_FIELDS;
            }
            auto command_line_cl =
                    utils::extract_pg_keys_from_hdr(reader.header(), {"CL"}, "ID", "basecaller");
            // If dorado was run with --estimate-poly-a option, output polyA related fields in the summary
            if (command_line_cl["CL"].find("estimate-poly-a") != std::string::npos) {
                flags |= SummaryFileWriter::POLYA_FIELDS;
            }

            SamHdrPtr shared_hdr(sam_hdr_dup(reader.header()));
            auto hdr = shared_hdr.get();
            int num_rg_lines = sam_hdr_count_lines(hdr, "RG");
            KString tag_wrapper(100000);
            auto& tag_value = tag_wrapper.get();
            for (int i = 0; i < num_rg_lines; ++i) {
                if (sam_hdr_find_tag_pos(hdr, "RG", i, "SM", &tag_value) == 0) {
                    flags |= SummaryFileWriter::BARCODING_FIELDS;
                    break;
                }
            }
        }

        DORADO_LOG(info, "Preprocessing complete");
    }

    return {flags, std::move(alignment_counts)};
}

}  // namespace dorado::cli
