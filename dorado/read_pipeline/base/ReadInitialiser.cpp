#include "read_pipeline/base/ReadInitialiser.h"

#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>

#include <iomanip>
#include <sstream>

namespace {

int get_min_qscore(sam_hdr_t* header) {
    auto command_line_cl =
            dorado::utils::extract_pg_keys_from_hdr(header, {"CL"}, "ID", "basecaller");
    // If dorado was run with --min-qscore option, parse the value so we can re-evaluate the pass/fail criterion
    std::stringstream cl{command_line_cl["CL"]};
    std::string out;
    while (cl.good()) {
        cl >> std::quoted(out);
        if (out == "--min-qscore") {
            cl >> std::quoted(out);
            return std::atoi(out.c_str());
        }
    }
    return 0;
}

}  // namespace

namespace dorado {

ReadInitialiser::ReadInitialiser(sam_hdr_t* hdr, AlignmentCounts aln_counts)
        : m_header(hdr),
          m_alignment_counts(std::move(aln_counts)),
          m_read_groups(utils::parse_read_groups(m_header)),
          m_minimum_qscore(get_min_qscore(m_header)) {}

void ReadInitialiser::update_read_attributes(HtsData& data) const {
    if (const auto rg_tag = bam_aux_get(data.bam_ptr.get(), "RG"); rg_tag != nullptr) {
        const std::string rg_tag_value = bam_aux2Z(rg_tag);
        const auto read_group_it = m_read_groups.find(rg_tag_value);
        if (read_group_it == std::cend(m_read_groups)) {
            return;
        }
        const auto& read_group = read_group_it->second;
        data.read_attrs.model_stride = read_group.model_stride;
        data.read_attrs.experiment_id = read_group.experiment_id;
        data.read_attrs.sample_id = read_group.sample_id;
        // position_id is not currently stored in the output files
        // data.read_attrs.position_id = read_group.position_id;
        data.read_attrs.flowcell_id = read_group.flowcell_id;
        data.read_attrs.protocol_run_id = read_group.run_id;
        data.read_attrs.protocol_start_time_ms =
                utils::get_unix_time_ms_from_string_timestamp(read_group.exp_start_time);

        if (const auto qs_tag = bam_aux_get(data.bam_ptr.get(), "qs"); qs_tag != nullptr) {
            const float qscore = static_cast<float>(bam_aux2f(qs_tag));
            data.read_attrs.is_status_pass = qscore >= m_minimum_qscore;
        }

        try {
            if (const auto st_tag = bam_aux_get(data.bam_ptr.get(), "st"); st_tag != nullptr) {
                const std::string read_start_time_str = bam_aux2Z(st_tag);
                const auto acq_start_time =
                        utils::get_unix_time_ms_from_string_timestamp(read_group.acq_start_time);
                const auto read_start_time =
                        utils::get_unix_time_ms_from_string_timestamp(read_start_time_str);
                data.read_attrs.start_time_ms = read_start_time - acq_start_time;
            }
        } catch (...) {
            // can't parse something, ignore start_time and continue
        }
    }
}

void ReadInitialiser::update_barcoding_fields(HtsData& data) const {
    if (const auto rg_tag = bam_aux_get(data.bam_ptr.get(), "RG"); rg_tag != nullptr) {
        const std::string rg_tag_value = bam_aux2Z(rg_tag);
        KString ks_wrapper(100000);
        auto& ks = ks_wrapper.get();

        auto barcoding_result = std::make_shared<BarcodeScoreResult>();
        bool found = false;
        if (sam_hdr_find_tag_id(m_header, "RG", "ID", rg_tag_value.c_str(), "SM", &ks) == 0) {
            barcoding_result->barcode_name = std::string(ks.s, ks.l);
            found = true;
        }
        if (sam_hdr_find_tag_id(m_header, "RG", "ID", rg_tag_value.c_str(), "al", &ks) == 0) {
            barcoding_result->alias = std::string(ks.s, ks.l);
            found = true;
        }
        if (sam_hdr_find_tag_id(m_header, "RG", "ID", rg_tag_value.c_str(), "bk", &ks) == 0) {
            barcoding_result->kit = std::string(ks.s, ks.l);
            found = true;
        }
        if (found) {
            data.barcoding_result = std::move(barcoding_result);
        }
    }
}

void ReadInitialiser::update_alignment_fields(HtsData& data) const {
    const auto alignment_counts_it = m_alignment_counts.find(bam_get_qname(data.bam_ptr.get()));
    if (alignment_counts_it != std::end(m_alignment_counts)) {
        const auto& counts = alignment_counts_it->second;
        data.read_attrs.num_alignments = counts[0];
        data.read_attrs.num_secondary_alignments = counts[1];
        data.read_attrs.num_supplementary_alignments = counts[2];
    }
}

void update_alignment_counts(const std::filesystem::path& path, AlignmentCounts& alignment_counts) {
    const auto file = dorado::HtsFilePtr(hts_open(path.string().c_str(), "r"));
    if (file->format.format != htsExactFormat::sam && file->format.format != htsExactFormat::bam) {
        return;
    }

    dorado::SamHdrPtr header(sam_hdr_read(file.get()));
    if (header->n_targets == 0) {
        return;
    }

    BamPtr record(bam_init1());
    while (sam_read1(file.get(), header.get(), record.get()) >= 0) {
        if (record->core.flag & BAM_FUNMAP) {
            continue;
        }
        auto& read_counts = alignment_counts[bam_get_qname(record.get())];
        if (record->core.flag & BAM_FSUPPLEMENTARY) {
            ++read_counts[2];
        }
        if (record->core.flag & BAM_FSECONDARY) {
            ++read_counts[1];
        }
        ++read_counts[0];
    }
}

}  // namespace dorado
