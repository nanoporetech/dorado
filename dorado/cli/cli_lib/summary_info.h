#pragma once

#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/SummaryFileWriter.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadInitialiser.h"

#include <filesystem>
#include <tuple>
#include <vector>

namespace dorado {

namespace cli {

inline std::tuple<hts_writer::SummaryFileWriter::FieldFlags, AlignmentCounts> make_summary_info(
        const std::vector<std::filesystem::path>& all_files) {
    using namespace hts_writer;
    SummaryFileWriter::FieldFlags flags =
            SummaryFileWriter::BASECALLING_FIELDS | SummaryFileWriter::EXPERIMENT_FIELDS;
    AlignmentCounts alignment_counts;
    if (!(all_files.size() == 1 && all_files[0] == "-")) {
        for (const auto& input_file : all_files) {
            update_alignment_counts(input_file, alignment_counts);
            HtsReader reader(input_file.string(), std::nullopt);
            if (reader.is_aligned) {
                flags |= SummaryFileWriter::ALIGNMENT_FIELDS;
            }
            auto command_line_cl =
                    utils::extract_pg_keys_from_hdr(reader.header(), {"CL"}, "ID", "basecaller");
            // If dorado was run with --estimate-poly-a option, output polyA related fields in the summary
            if (command_line_cl["CL"].find("estimate-poly-a") != std::string::npos) {
                flags |= SummaryFileWriter::POLYA_FIELDS;
            }

            SamHdrSharedPtr shared_hdr(sam_hdr_dup(reader.header()));
            auto hdr = const_cast<sam_hdr_t*>(shared_hdr.get());
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
    }

    return {flags, alignment_counts};
}

}  // namespace cli

}  // namespace dorado