#include "bam_utils.h"

#include "SampleSheet.h"
#include "barcode_kits.h"
#include "sequence_utils.h"

#include <htslib/sam.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>

#ifdef _WIN32
// seq_nt16_str is referred to in the hts-3.lib stub on windows, but has not been declared dllimport for
//  client code, so it comes up as an undefined reference when linking the stub.
const char seq_nt16_str[] = "=ACMGRSVTWYHKDBN";
#endif  // _WIN32

namespace dorado::utils {

namespace {

// Convert the 4bit encoded sequence in a bam1_t structure
// into a string.
std::string convert_nt16_to_str(uint8_t* bseq, size_t slen) {
    std::string seq(slen, '*');
    for (size_t i = 0; i < slen; i++) {
        seq[i] = seq_nt16_str[bam_seqi(bseq, i)];
    }
    return seq;
}

void emit_read_group(sam_hdr_t* hdr,
                     const std::string& read_group_line,
                     const std::string& id,
                     const std::string& additional_tags) {
    auto line = "@RG\tID:" + id + '\t' + read_group_line + additional_tags + '\n';
    sam_hdr_add_lines(hdr, line.c_str(), 0);
}

std::string read_group_to_string(const dorado::ReadGroup& read_group) {
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
           << (read_group.modbase_models.empty() ? ""
                                                 : (" modbase_models=" + read_group.modbase_models))
           << " runid=" << value_or_unknown(read_group.run_id) << "\t";
        rg << "LB:" << value_or_unknown(read_group.sample_id) << "\t";
        rg << "SM:" << value_or_unknown(read_group.sample_id);
    }
    return rg.str();
}

void add_barcode_kit_rg_hdrs(sam_hdr_t* hdr,
                             const std::unordered_map<std::string, ReadGroup>& read_groups,
                             const std::string& kit_name,
                             const utils::SampleSheet* const sample_sheet) {
    auto get_barcode_sequence =
            [barcode_sequences = barcode_kits::get_barcodes()](const std::string& barcode_name) {
                auto sequence_itr = barcode_sequences.find(barcode_name);
                if (sequence_itr != barcode_sequences.end()) {
                    return sequence_itr->second;
                }
                throw std::runtime_error("Unrecognised barcode name: " + barcode_name);
            };

    const auto& kit_info_map = barcode_kits::get_kit_infos();
    auto kit_info = kit_info_map.find(kit_name);
    if (kit_info == kit_info_map.end()) {
        throw std::runtime_error("Unrecognised kit name: " + kit_name);
    }
    for (const auto& barcode_name : kit_info->second.barcodes) {
        const auto additional_tags = "\tBC:" + get_barcode_sequence(barcode_name);
        const auto normalized_barcode_name = barcode_kits::normalize_barcode_name(barcode_name);
        for (const auto& read_group : read_groups) {
            std::string alias;
            auto id = read_group.first + '_';
            if (sample_sheet) {
                if (!sample_sheet->barcode_is_permitted(normalized_barcode_name)) {
                    continue;
                }

                alias = sample_sheet->get_alias(
                        read_group.second.flowcell_id, read_group.second.position_id,
                        read_group.second.experiment_id, normalized_barcode_name);
            }
            if (!alias.empty()) {
                id += alias;
            } else {
                id += barcode_kits::generate_standard_barcode_name(kit_name, barcode_name);
            }
            const std::string read_group_tags = read_group_to_string(read_group.second);
            emit_read_group(hdr, read_group_tags, id, additional_tags);
        }
    }
}

}  // namespace

bool try_add_fastq_header_tag(bam1_t* record, const std::string& header) {
    // validate the fastq header contains only printable characters including SPACE
    // i.e. ' ' (0x20) through to '~' (0x7E)
    // Note this will not write an HtsLib generated fastq header line which has the bam tags appended
    if (std::any_of(header.begin(), header.end(), [](char c) { return c < 0x20 || c > 0x7e; })) {
        return false;
    }

    return bam_aux_append(record, "fq", 'Z', static_cast<int>(header.size() + 1),
                          reinterpret_cast<const uint8_t*>(header.c_str())) == 0;
}

int remove_fastq_header_tag(bam1_t* record) {
    auto tag_ptr = bam_aux_get(record, "fq");
    if (!tag_ptr) {
        return 0;  // return success result for bam_aux_del, as we assume it wasn't present.
    }
    return bam_aux_del(record, tag_ptr);
}

void add_hd_header_line(sam_hdr_t* hdr) {
    sam_hdr_add_line(hdr, "HD", "VN", SAM_FORMAT_VERSION, "SO", "unknown", nullptr);
}

void add_rg_headers(sam_hdr_t* hdr, const std::unordered_map<std::string, ReadGroup>& read_groups) {
    for (const auto& read_group : read_groups) {
        const std::string read_group_tags = read_group_to_string(read_group.second);
        emit_read_group(hdr, read_group_tags, read_group.first, {});
    }
}

void add_rg_headers_with_barcode_kit(sam_hdr_t* hdr,
                                     const std::unordered_map<std::string, ReadGroup>& read_groups,
                                     const std::string& kit_name,
                                     const utils::SampleSheet* const sample_sheet) {
    add_rg_headers(hdr, read_groups);
    add_barcode_kit_rg_hdrs(hdr, read_groups, kit_name, sample_sheet);
}

void add_sq_hdr(sam_hdr_t* hdr, const sq_t& seqs) {
    for (const auto& pair : seqs) {
        sam_hdr_add_line(hdr, "SQ", "SN", pair.first.c_str(), "LN",
                         std::to_string(pair.second).c_str(), NULL);
    }
}

void strip_alignment_data_from_header(sam_hdr_t* hdr) {
    sam_hdr_remove_except(hdr, "SQ", nullptr, nullptr);
    sam_hdr_change_HD(hdr, "SO", "unknown");
}

std::map<std::string, std::string> get_read_group_info(sam_hdr_t* header, const char* key) {
    if (header == nullptr) {
        throw std::invalid_argument("header cannot be nullptr");
    }

    int num_read_groups = sam_hdr_count_lines(header, "RG");
    if (num_read_groups == -1) {
        throw std::runtime_error("no read groups in file");
    }

    KString rg_wrapper(1000000);
    auto rg = rg_wrapper.get();
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

std::map<std::string, std::string> extract_pg_keys_from_hdr(const std::string& filename,
                                                            const std::vector<std::string>& keys,
                                                            const char* ID_key,
                                                            const char* ID_val) {
    std::map<std::string, std::string> pg_keys;
    auto file = HtsFilePtr(hts_open(filename.c_str(), "r"));
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    SamHdrPtr header(sam_hdr_read(file.get()));
    if (!header) {
        throw std::runtime_error("Could not open header from file: " + filename);
    }
    return extract_pg_keys_from_hdr(header.get(), keys, ID_key, ID_val);
}

std::map<std::string, std::string> extract_pg_keys_from_hdr(sam_hdr_t* header,
                                                            const std::vector<std::string>& keys,
                                                            const char* ID_key,
                                                            const char* ID_val) {
    std::map<std::string, std::string> pg_keys;
    if (!header) {
        throw std::runtime_error("Provided header cannot be a nullptr.");
    }
    KString val_wrapper(1000000);
    auto val = val_wrapper.get();
    for (auto& k : keys) {
        auto ret = sam_hdr_find_tag_id(header, "PG", ID_key, ID_val, k.c_str(), &val);
        if (ret == 0) {
            pg_keys[k] = std::string(val.s);
        }
    }
    return pg_keys;
}

std::string extract_sequence(bam1_t* input_record) {
    auto bseq = bam_get_seq(input_record);
    int seqlen = input_record->core.l_qseq;
    std::string seq = convert_nt16_to_str(bseq, seqlen);
    return seq;
}

std::vector<uint8_t> extract_quality(bam1_t* input_record) {
    auto qual_aln = bam_get_qual(input_record);
    int seqlen = input_record->core.l_qseq;
    std::vector<uint8_t> qual;
    if (qual_aln) {
        qual = std::vector<uint8_t>(bam_get_qual(input_record),
                                    bam_get_qual(input_record) + seqlen);
    }
    return qual;
}

std::tuple<int, std::vector<uint8_t>> extract_move_table(bam1_t* input_record) {
    auto move_vals_aux = bam_aux_get(input_record, "mv");
    std::vector<uint8_t> move_vals;
    int stride = 0;
    if (move_vals_aux) {
        int len = bam_auxB_len(move_vals_aux);
        // First element for move table array is the stride.
        stride = int(bam_auxB2i(move_vals_aux, 0));
        move_vals.resize(len - 1);
        for (int i = 1; i < len; i++) {
            move_vals[i - 1] = uint8_t(bam_auxB2i(move_vals_aux, i));
        }
    }
    return {stride, move_vals};
}

std::tuple<std::string, std::vector<uint8_t>> extract_modbase_info(bam1_t* input_record) {
    std::string modbase_str;
    std::vector<uint8_t> modbase_probs;
    auto modbase_str_aux = bam_aux_get(input_record, "MM");
    if (modbase_str_aux) {
        modbase_str = std::string(bam_aux2Z(modbase_str_aux));

        auto modbase_prob_aux = bam_aux_get(input_record, "ML");
        int len = bam_auxB_len(modbase_prob_aux);
        modbase_probs.resize(len);
        for (int i = 0; i < len; i++) {
            modbase_probs[i] = uint8_t(bam_auxB2i(modbase_prob_aux, i));
        }
    }

    return {modbase_str, modbase_probs};
}

bool validate_bam_tag_code(const std::string& bam_name) {
    // Check the supplied bam_name is a single character
    if (bam_name.size() == 1 && std::isalpha(static_cast<unsigned char>(bam_name[0]))) {
        return true;
    }

    // Check the supplied bam_name is a simple integer and if so, assume it's a CHEBI code.
    if (std::all_of(bam_name.begin(), bam_name.end(),
                    [](const char& c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
        return true;
    }
    return false;
}

std::vector<uint32_t> trim_cigar(uint32_t n_cigar,
                                 const uint32_t* cigar,
                                 const std::pair<int, int>& trim_interval) {
    std::vector<uint32_t> ops;
    ops.reserve(n_cigar);

    auto trim_s = trim_interval.first;
    auto trim_e = trim_interval.second;
    auto trim_len = trim_e - trim_s;

    int cursor = 0;
    // The in_interval flag tracks where in the sequence
    // the ops lie, i.e. either outside the interval or
    // inside the interval.
    bool in_interval = false;
    for (uint32_t i = 0; i < n_cigar; i++) {
        const uint32_t op = bam_cigar_op(cigar[i]);
        const uint32_t oplen = bam_cigar_oplen(cigar[i]);
        // According to htslib docs, bit 1 represents "consumes
        // query" and bit 2 represents "consumes reference"
        auto type = std::bitset<2>(bam_cigar_type(op));
        if (type[0]) {
            // consumes query positions
            cursor += oplen;
        }
        if (cursor > trim_e) {
            uint32_t new_len;
            // In case the trim interval completely consumed
            // by a CIGAR op, the state machine never enters
            // the in_interval state. In that scenario, the
            // final CIGAR string is the last CIGAR op but
            // with the length of the interval.
            // Otherwise the last op length is only the portion
            // of the interval that overlaps with CIGAR op.
            if (!in_interval) {
                new_len = trim_len;
            } else {
                new_len = trim_e - (cursor - oplen);
            }
            // If the overlap of the last op and the remianing
            // interval is 0 (i.e. the interval was covered with the
            // previos op), don't add anything.
            if (new_len > 0) {
                ops.push_back(new_len << BAM_CIGAR_SHIFT | op);
            }
            break;
        } else if (cursor > trim_s && !in_interval) {
            // If the op straddles the start boundary of the interval,
            // retain the overlap portion and switch state machine to
            // being in_interval.
            in_interval = true;
            ops.push_back((cursor - trim_s) << BAM_CIGAR_SHIFT | op);
        } else if (in_interval) {
            // If the op is inside the interval and not at boundaries,
            // retain the op.
            ops.push_back(cigar[i]);
        }
    }

    // Remove any ops from the end that don't move the query cursor.
    // Because the cursor isn't updated for ops that affect the
    // reference only, ops such as DELETE can be left around which
    // need to be cleaned up.
    int last_pos = int(ops.size() - 1);
    for (; last_pos > 0; last_pos--) {
        const uint32_t op = bam_cigar_op(ops[last_pos]);
        auto type = std::bitset<2>(bam_cigar_type(op));
        if (type[0]) {
            break;
        }
    }
    ops.erase(ops.begin() + last_pos + 1, ops.end());

    return ops;
}

uint32_t ref_pos_consumed(uint32_t n_cigar, const uint32_t* cigar, uint32_t query_pos) {
    // In this algorithm the reference and query cursor are both
    // consumed based on the type of op encountered in the CIGAR
    // string. Once the query cursor reached the desired
    // query position, the reference cursor position at that point
    // is returned.
    uint32_t query_cursor = 0;
    uint32_t ref_cursor = 0;
    for (uint32_t i = 0; i < n_cigar; i++) {
        const uint32_t op = bam_cigar_op(cigar[i]);
        const uint32_t oplen = bam_cigar_oplen(cigar[i]);
        auto type = std::bitset<2>(bam_cigar_type(op));

        if (type[0] && !type[1]) {
            // Ops that consume query only.
            query_cursor += oplen;
            if (query_cursor >= query_pos) {
                break;
            }
        } else if (!type[0] && type[1]) {
            // Ops that consume reference only.
            ref_cursor += oplen;
        } else if (type[0] && type[1]) {
            // Ops that consume query & reference.
            if (query_cursor + oplen > query_pos) {
                ref_cursor += (query_pos - query_cursor);
                break;
            } else {
                ref_cursor += oplen;
                query_cursor += oplen;
            }
        }
    }
    return ref_cursor;
}

std::string cigar2str(uint32_t n_cigar, const uint32_t* cigar) {
    std::string cigar_str = "";
    for (uint32_t i = 0; i < n_cigar; i++) {
        auto oplen = bam_cigar_oplen(cigar[i]);
        auto opchr = bam_cigar_opchr(cigar[i]);
        cigar_str += std::to_string(oplen) + std::string(1, opchr);
    }
    return cigar_str;
}

BamPtr new_unmapped_record(bam1_t* input_record, std::string seq, std::vector<uint8_t> qual) {
    if (seq.empty()) {
        seq = extract_sequence(input_record);
        qual = extract_quality(input_record);
        if (bam_is_rev(input_record)) {
            seq = utils::reverse_complement(seq);
            std::reverse(qual.begin(), qual.end());
        }
    }

    bam1_t* out_record = bam_init1();
    bam_set1(out_record, input_record->core.l_qname - input_record->core.l_extranul - 1,
             bam_get_qname(input_record), 4 /*flag*/, -1 /*tid*/, -1 /*pos*/, 0 /*mapq*/,
             0 /*n_cigar*/, nullptr /*cigar*/, -1 /*mtid*/, -1 /*mpos*/, 0 /*isize*/, seq.size(),
             seq.data(), qual.empty() ? nullptr : (char*)qual.data(), bam_get_l_aux(input_record));
    memcpy(bam_get_aux(out_record), bam_get_aux(input_record), bam_get_l_aux(input_record));
    out_record->l_data += bam_get_l_aux(input_record);
    remove_alignment_tags_from_record(out_record);
    return BamPtr(out_record);
}

void remove_alignment_tags_from_record(bam1_t* record) {
    // Iterate through all tags and check against known set
    // of tags to remove.
    static const std::set<std::string> tags_to_remove = {"SA", "NM", "ms", "AS", "nn", "de",
                                                         "dv", "tp", "cm", "s1", "s2", "MD",
                                                         "zd", "rl", "bh", "cs", "TS"};

    uint8_t* aux_ptr = bam_aux_first(record);
    while (aux_ptr != NULL) {
        auto tag_ptr = bam_aux_tag(aux_ptr);
        std::string tag = std::string(tag_ptr, tag_ptr + 2);
        if (tags_to_remove.find(tag) != tags_to_remove.end()) {
            aux_ptr = bam_aux_remove(record, aux_ptr);
        } else {
            aux_ptr = bam_aux_next(record, aux_ptr);
        }
    }
}

}  // namespace dorado::utils
