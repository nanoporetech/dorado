#include "read_pipeline/base/ReadInitialiser.h"

#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>

#include <iomanip>
#include <sstream>

namespace dorado {

namespace {

void assign_not_empty(std::string& attr_target, const std::string_view maybe_value) {
    if (!maybe_value.empty()) {
        attr_target = maybe_value;
    }
};

int get_min_qscore(sam_hdr_t* header) {
    auto command_line_cl = utils::extract_pg_keys_from_hdr(header, {"CL"}, "ID", "basecaller");
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

ReadInitialiser::ReadInitialiser(sam_hdr_t* hdr, AlignmentCounts aln_counts, TrimFlags trim_flags)
        : m_header(hdr),
          m_alignment_counts(std::move(aln_counts)),
          m_read_groups(utils::parse_read_groups(m_header)),
          m_minimum_qscore(get_min_qscore(m_header)),
          m_trim_flags(trim_flags) {}

void ReadInitialiser::update_read_attributes(HtsData& data) const {
    if (const auto tm_tag = bam_aux_get(data.bam_ptr.get(), "tm"); tm_tag != nullptr) {
        // update trim flags from the record itself
        std::string_view tm_str = bam_aux2Z(tm_tag);
        data.read_attrs.trim_flags = TrimFlags::from_string(tm_str);
        data.read_attrs.trim_flags.merge(m_trim_flags);
    }

    if (const auto rg_tag = bam_aux_get(data.bam_ptr.get(), "RG"); rg_tag != nullptr) {
        const std::string rg_tag_value = bam_aux2Z(rg_tag);
        const auto read_group_it = m_read_groups.find(rg_tag_value);
        if (read_group_it == std::cend(m_read_groups)) {
            return;
        }
        const auto& read_group = read_group_it->second;
        data.read_attrs.model_stride = read_group.model_stride;
        assign_not_empty(data.read_attrs.flowcell_id, read_group.flowcell_id);
        assign_not_empty(data.read_attrs.experiment_id, read_group.experiment_id);
        assign_not_empty(data.read_attrs.sample_id, read_group.sample_id);
        assign_not_empty(data.read_attrs.protocol_run_id, read_group.run_id);

        data.read_attrs.trim_flags = read_group.trim_flags;
        data.read_attrs.trim_flags.merge(m_trim_flags);

        // position_id is not currently stored in the output files
        // assign_not_empty(data.read_attrs.position_id, read_group.position_id);
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
            barcoding_result->normalized_barcode_name =
                    barcode_kits::normalize_barcode_name(barcoding_result->barcode_name);
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

void ReadInitialiser::undemux_read_group(HtsData& data) const {
    std::string alias;
    bam1_t* record = data.bam_ptr.get();
    if (const auto al_tag = bam_aux_get(record, "al"); al_tag != nullptr) {
        alias = bam_aux2Z(al_tag);
        bam_aux_del(record, al_tag);
        bam_aux_update_str(record, "BC", UNCLASSIFIED_STR.length() + 1, UNCLASSIFIED_STR.c_str());
    } else if (const auto bc_tag = bam_aux_get(record, "BC"); bc_tag != nullptr) {
        alias = bam_aux2Z(bc_tag);
        bam_aux_update_str(record, "BC", UNCLASSIFIED_STR.length() + 1, UNCLASSIFIED_STR.c_str());
    }

    if (const auto rg_tag = bam_aux_get(record, "RG"); rg_tag != nullptr) {
        std::string rg_tag_value = bam_aux2Z(rg_tag);
        if (auto index = rg_tag_value.find(alias); index != rg_tag_value.npos && index != 0) {
            rg_tag_value = rg_tag_value.substr(0, index - 1);
            bam_aux_update_str(record, "RG", index, rg_tag_value.c_str());
        }
    }
}

}  // namespace dorado
