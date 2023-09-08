#include "bam_utils.h"

#include "barcode_kits.h"
#include "sequence_utils.h"

#include <htslib/sam.h>

#include <cctype>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>

namespace dorado::utils {

void add_rg_hdr(sam_hdr_t* hdr,
                const std::unordered_map<std::string, ReadGroup>& read_groups,
                const std::vector<std::string>& barcode_kits) {
    const auto& barcode_kit_infos = barcode_kits::get_kit_infos();
    const auto& barcode_sequences = barcode_kits::get_barcodes();

    // Convert a ReadGroup to a string
    auto to_string = [](const ReadGroup& read_group) {
        // Lambda function to return "unknown" if string is empty
        auto value_or_unknown = [](std::string_view s) { return s.empty() ? "unknown" : s; };
        std::ostringstream rg;
        {
            rg << "PU:" << value_or_unknown(read_group.flowcell_id) << "\t";
            rg << "PM:" << value_or_unknown(read_group.device_id) << "\t";
            rg << "DT:" << value_or_unknown(read_group.exp_start_time) << "\t";
            rg << "PL:"
               << "ONT"
               << "\t";
            rg << "DS:"
               << "basecall_model=" << value_or_unknown(read_group.basecalling_model)
               << " runid=" << value_or_unknown(read_group.run_id) << "\t";
            rg << "LB:" << value_or_unknown(read_group.sample_id) << "\t";
            rg << "SM:" << value_or_unknown(read_group.sample_id);
        }
        return std::move(rg).str();
    };

    auto emit_read_group = [hdr](const std::string& read_group_line, const std::string& id,
                                 const std::string& additional_tags) {
        auto line = "@RG\tID:" + id + '\t' + read_group_line + additional_tags + '\n';
        sam_hdr_add_lines(hdr, line.c_str(), 0);
    };

    // Emit read group headers without a barcode arrangement.
    for (const auto& read_group : read_groups) {
        const std::string read_group_tags = to_string(read_group.second);
        emit_read_group(read_group_tags, read_group.first, {});
    }

    // Emit read group headers for each barcode arrangement.
    for (const auto& kit_name : barcode_kits) {
        const auto& kit_info = barcode_kit_infos.at(kit_name);
        for (const auto& barcode_name : kit_info.barcodes) {
            const auto additional_tags = "\tBC:" + barcode_sequences.at(barcode_name);
            for (const auto& read_group : read_groups) {
                auto id = read_group.first + '_' + kit_name + '_' + barcode_name;
                const std::string read_group_tags = to_string(read_group.second);
                emit_read_group(read_group_tags, id, additional_tags);
            }
        }
    }
}

void add_sq_hdr(sam_hdr_t* hdr, const sq_t& seqs) {
    for (auto pair : seqs) {
        char* name;
        int length;
        std::tie(name, length) = pair;
        sam_hdr_add_line(hdr, "SQ", "SN", name, "LN", std::to_string(length).c_str(), NULL);
    }
}

std::map<std::string, std::string> get_read_group_info(sam_hdr_t* header, const char* key) {
    if (header == nullptr) {
        throw std::invalid_argument("header cannot be nullptr");
    }

    int num_read_groups = sam_hdr_count_lines(header, "RG");
    if (num_read_groups == -1) {
        throw std::runtime_error("no read groups in file");
    }

    kstring_t rg = {0, 0, NULL};
    std::map<std::string, std::string> read_group_info;

    for (int i = 0; i < num_read_groups; ++i) {
        const char* id = sam_hdr_line_name(header, "RG", i);
        if (id == nullptr) {
            continue;
        }

        std::string read_group_id(id);
        int res = sam_hdr_find_tag_id(header, "RG", "ID", id, key, &rg);
        if (res == 0 && rg.l > 0) {
            read_group_info[read_group_id] = std::string(rg.s, rg.l);
        }
    }

    free(rg.s);
    return read_group_info;
}

AlignmentOps get_alignment_op_counts(bam1_t* record) {
    AlignmentOps counts = {};

    uint32_t* cigar = bam_get_cigar(record);
    int n_cigar = record->core.n_cigar;

    if (bam_cigar_op(cigar[0]) == BAM_CSOFT_CLIP) {
        counts.softclip_start = bam_cigar_oplen(cigar[0]);
    }

    if (bam_cigar_op(cigar[n_cigar - 1]) == BAM_CSOFT_CLIP) {
        counts.softclip_end = bam_cigar_oplen(cigar[n_cigar - 1]);
    }

    for (int i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]);
        int op_len = bam_cigar_oplen(cigar[i]);

        switch (op) {
        case BAM_CMATCH:
            counts.matches += op_len;
            break;
        case BAM_CINS:
            counts.insertions += op_len;
            break;
        case BAM_CDEL:
            counts.deletions += op_len;
            break;
        default:
            break;
        }
    }

    uint8_t* md_ptr = bam_aux_get(record, "MD");

    if (md_ptr) {
        int i = 0;
        int md_length = 0;
        char* md = bam_aux2Z(md_ptr);

        while (md[i]) {
            if (std::isdigit(md[i])) {
                md_length = md_length * 10 + (md[i] - '0');
            } else {
                if (md[i] == '^') {
                    // Skip deletions
                    i++;
                    while (md[i] && !std::isdigit(md[i])) {
                        i++;
                    }
                } else {
                    // Substitution found
                    counts.substitutions++;
                    md_length++;
                }
            }
            i++;
        }
    }

    return counts;
}

std::map<std::string, std::string> extract_pg_keys_from_hdr(const std::string filename,
                                                            const std::vector<std::string>& keys) {
    std::map<std::string, std::string> pg_keys;
    auto file = hts_open(filename.c_str(), "r");
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    SamHdrPtr header(sam_hdr_read(file));
    if (!header) {
        throw std::runtime_error("Could not open header from file: " + filename);
    }
    kstring_t val = {0, 0, NULL};
    // The kstring is pre-allocated with 1MB of space to work around a Windows DLL
    // cross-heap segmentation fault. The ks_resize/ks_free functions from htslib
    // are inline functions. When htslib is shipped as a DLL, some of these functions
    // are inlined into the DLL code through other htslib APIs. But those same functions
    // also get inlined into dorado code when invoked directly. As a result, it's possible
    // that an htslib APIs resizes a string using the DLL code. But when a ks_free
    // is attempted on it from dorado, there's cross-heap behavior and a segfault occurs.
    ks_resize(&val, 1e6);
    for (auto& k : keys) {
        auto ret = sam_hdr_find_tag_id(header.get(), "PG", NULL, NULL, k.c_str(), &val);
        if (ret != 0) {
            throw std::runtime_error("Required key " + k + " not found in header of " + filename);
        }
        pg_keys[k] = std::string(val.s);
    }
    ks_free(&val);
    hts_close(file);
    return pg_keys;
}

std::string extract_sequence(bam1_t* input_record, int seqlen) {
    auto bseq = bam_get_seq(input_record);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
    return seq;
}

std::vector<uint8_t> extract_quality(bam1_t* input_record, int seqlen) {
    auto qual_aln = bam_get_qual(input_record);
    std::vector<uint8_t> qual;
    if (qual_aln) {
        qual = std::vector<uint8_t>(bam_get_qual(input_record),
                                    bam_get_qual(input_record) + seqlen);
    }
    return qual;
}

std::vector<uint8_t> extract_move_table(bam1_t* input_record) {
    auto move_vals_aux = bam_aux_get(input_record, "mv");
    std::vector<uint8_t> move_vals;
    if (move_vals_aux) {
        int len = bam_auxB_len(move_vals_aux);
        move_vals.resize(len);
        for (int i = 0; i < len; i++) {
            move_vals[i] = bam_auxB2i(move_vals_aux, i);
        }
        bam_aux_del(input_record, move_vals_aux);
    }
    return move_vals;
}

std::tuple<std::string, std::vector<int8_t>> extract_modbase_info(bam1_t* input_record) {
    std::string modbase_str;
    std::vector<int8_t> modbase_probs;
    auto modbase_str_aux = bam_aux_get(input_record, "MM");
    if (modbase_str_aux) {
        modbase_str = std::string(bam_aux2Z(modbase_str_aux));
        bam_aux_del(input_record, modbase_str_aux);

        auto modbase_prob_aux = bam_aux_get(input_record, "ML");
        int len = bam_auxB_len(modbase_prob_aux);
        modbase_probs.resize(len);
        for (int i = 0; i < len; i++) {
            modbase_probs[i] = bam_auxB2i(modbase_prob_aux, i);
        }
        bam_aux_del(input_record, modbase_prob_aux);
    }

    return {modbase_str, modbase_probs};
}

}  // namespace dorado::utils
