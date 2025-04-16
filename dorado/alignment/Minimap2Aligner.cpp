#include "Minimap2Aligner.h"

#include "sam_utils.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <minimap.h>

#include <stdexcept>

//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include <mmpriv.h>

namespace {
// If an alignment has secondary alignments, add that information
// to each record. Follows minimap2 conventions.
void add_sa_tag(bam1_t* record,
                const mm_reg1_t* regs,
                int32_t hits,
                int32_t aln_idx,
                int32_t l_seq,
                const mm_idx_t* idx) {
    std::stringstream ss;
    for (int i = 0; i < hits; i++) {
        if (i == aln_idx) {
            continue;
        }
        const mm_reg1_t* r = &regs[i];

        if (r->parent != r->id || r->p == 0) {
            continue;
        }

        int num_matches = 0, num_inserts = 0, num_deletes = 0;
        int clip3 = 0, clip5 = 0;

        if (r->qe - r->qs < r->re - r->rs) {
            num_matches = r->qe - r->qs;
            num_deletes = (r->re - r->rs) - num_matches;
        } else {
            num_matches = r->re - r->rs;
            num_inserts = (r->qe - r->qs) - num_matches;
        }

        clip5 = r->rev ? l_seq - r->qe : r->qs;
        clip3 = r->rev ? r->qs : l_seq - r->qe;

        ss << std::string(idx->seq[r->rid].name) << ",";
        ss << r->rs + 1 << ",";
        ss << "+-"[r->rev] << ",";
        if (clip5) {
            ss << clip5 << "S";
        }
        if (num_matches) {
            ss << num_matches << "M";
        }
        if (num_inserts) {
            ss << num_inserts << "I";
        }
        if (num_deletes) {
            ss << num_deletes << "D";
        }
        if (clip3) {
            ss << clip3 << "S";
        }
        ss << "," << r->mapq << "," << (r->blen - r->mlen + r->p->n_ambi) << ";";
    }
    std::string sa = ss.str();
    if (!sa.empty()) {
        bam_aux_append(record, "SA", 'Z', int(sa.length() + 1), (uint8_t*)sa.c_str());
    }
}

size_t find_next_tab(const std::string_view& str, size_t pos) {
    const auto p = str.find('\t', pos);
    if (p == std::string::npos) {
        throw std::runtime_error("Error parsing SAM string.");
    }
    return p;
}

void update_read_record(dorado::AlignmentResult& alignment, int new_mapq, bool first_block) {
    if (!first_block) {
        // All primary and supplemental alignments that aren't from the first block should
        // be marked as secondary alignments.
        alignment.supplementary_alignment = false;
        alignment.secondary_alignment = true;
    }
    const std::string_view sam_string(alignment.sam_string);
    std::string out;
    out.reserve(sam_string.size() + 3);  // flags field could potentially go from 1 digit to 4.

    // The FLAG field is the 2nd field.
    const size_t p1 = find_next_tab(sam_string, 0);
    out += sam_string.substr(0, p1 + 1);
    const size_t p2 = find_next_tab(sam_string, p1 + 1);
    if (first_block) {
        out += sam_string.substr(p1 + 1, p2 - p1 - 1);
    } else {
        const std::string flag_field(sam_string.substr(p1 + 1, p2 - p1 - 1));
        uint32_t flags = uint32_t(atoi(flag_field.c_str()));
        flags |= BAM_FSECONDARY;
        flags &= ~BAM_FSUPPLEMENTARY;
        out += std::to_string(flags);
    }
    if (new_mapq == -1) {
        // Keep the existing mapq value.
        out += sam_string.substr(p2);
    } else {
        // The MAPQ field is the 5th field.
        const size_t p3 = find_next_tab(sam_string, p2 + 1);
        const size_t p4 = find_next_tab(sam_string, p3 + 1);
        const size_t p5 = find_next_tab(sam_string, p4 + 1);
        out += sam_string.substr(p2, p4 - p2 + 1);
        out += std::to_string(new_mapq);
        out += sam_string.substr(p5);
    }
    alignment.sam_string.swap(out);
}

int compute_mapq(size_t num_alignments) {
    // Return -1 if there are less than 2 alignments. This means don't replace the mapq value.
    if (num_alignments < 2) {
        return -1;
    }
    // This gives -10*log(1 - 1/N), rounded to the nearest integer.
    static constexpr std::array<int32_t, 10> lookup{-1, -1, 3, 2, 1, 1, 1, 1, 1, 1};
    return (num_alignments >= 10) ? 0 : lookup[num_alignments];
}

}  // namespace

namespace dorado::alignment {

// Stripped of the prefix QNAME and postfix SEQ + \t + QUAL
const std::string UNMAPPED_SAM_LINE_STRIPPED{"\t4\t*\t0\t0\t*\t*\t0\t0\n"};

std::tuple<mm_reg1_t*, int> Minimap2Aligner::get_mapping(bam1_t* irecord, mm_tbuf_t* buf) {
    if (m_minimap_index->num_loaded_index_blocks() != 1) {
        throw std::logic_error(
                "Minimap2Aligner::get_mapping() called on fully-loaded split index.");
    }
    std::string_view qname(bam_get_qname(irecord));

    // get the sequence to map from the record
    std::string seq = utils::extract_sequence(irecord);

    // do the mapping
    int hits = 0;
    auto mm_index = m_minimap_index->index();
    const auto& mm_map_opts = m_minimap_index->mapping_options();
    mm_reg1_t* reg = mm_map(mm_index, static_cast<int>(seq.length()), seq.c_str(), &hits, buf,
                            &mm_map_opts, qname.data());
    return {reg, hits};
}

std::vector<BamPtr> Minimap2Aligner::align(bam1_t* irecord, mm_tbuf_t* buf) {
    // strip any existing alignment metadata from the read
    utils::remove_alignment_tags_from_record(irecord);

    int64_t best_score = 0;
    size_t best_index = 0;
    bool alignment_found = false;
    const size_t num_index_blocks = m_minimap_index->num_loaded_index_blocks();
    if (num_index_blocks == 0) {
        throw std::logic_error("Minimap2Aligner::align called without a loaded index.");
    }
    std::vector<std::vector<BamPtr>> block_records(num_index_blocks);
    for (size_t i = 0; i < num_index_blocks; ++i) {
        auto records = align_impl(irecord, buf, int(i));
        for (auto& record : records) {
            const uint16_t flags = record->core.flag;
            if ((flags & BAM_FUNMAP) == 0) {
                alignment_found = true;
            }
            if (((flags & BAM_FSECONDARY) == 0) && ((flags & BAM_FSUPPLEMENTARY) == 0)) {
                auto score_tag = bam_aux_get(record.get(), "AS");
                const int64_t score = (score_tag) ? bam_aux2i(score_tag) : 0;
                if (score > best_score) {
                    best_score = score;
                    best_index = i;
                }
            }
        }
        block_records[i].swap(records);
    }
    // Make the block with the best primary alignment the first one.
    if (best_index != 0) {
        block_records[0].swap(block_records[best_index]);
    }

    // Get rid of any spurious unmapped records.
    size_t non_sup_count = 0;  // Counts non-supplementary records, for mapq score recalculation.
    size_t num_blocks_mapped = 0;  // Counts blocks with mappings, for mapq score recalculation.
    if (alignment_found) {
        bool first_block = true;
        for (auto& records : block_records) {
            std::vector<BamPtr> filtered_records;
            for (auto& record : records) {
                if ((record->core.flag & BAM_FUNMAP) == 0) {
                    if (!first_block || ((record->core.flag & BAM_FSUPPLEMENTARY) == 0)) {
                        non_sup_count++;
                    }
                    filtered_records.emplace_back(std::move(record));
                }
            }
            records.swap(filtered_records);
            first_block = false;
            if (!records.empty()) {
                num_blocks_mapped++;
            }
        }
    } else {
        // Just keep the first block, which will only have one record.
        block_records.resize(1);
    }

    // Only recompute the mapq score if more than 1 block produced an alignment.
    const int new_mapq = (num_blocks_mapped > 1) ? compute_mapq(non_sup_count) : -1;

    // We can only have one primary alignment. Make all other primary alignments, and all supplementary
    // alignments that aren't in the first block, secondary.
    std::vector<BamPtr> final_records;
    const auto softclipping_on = (m_minimap_index->mapping_options().flag & MM_F_SOFTCLIP);
    bool first_block = true;
    for (auto& records : block_records) {
        for (auto& record : records) {
            if (!first_block) {
                record->core.flag |= BAM_FSECONDARY;
                record->core.flag &= ~BAM_FSUPPLEMENTARY;
                if (!softclipping_on) {
                    // Remove MM/ML/MN tags
                    if (auto tag = bam_aux_get(record.get(), "MM"); tag != nullptr) {
                        bam_aux_del(record.get(), tag);
                    }
                    if (auto tag = bam_aux_get(record.get(), "ML"); tag != nullptr) {
                        bam_aux_del(record.get(), tag);
                    }
                    if (auto tag = bam_aux_get(record.get(), "MN"); tag != nullptr) {
                        bam_aux_del(record.get(), tag);
                    }
                }
            }
            if (new_mapq != -1) {
                record->core.qual = uint8_t(new_mapq);
            }
            final_records.emplace_back(std::move(record));
        }
        first_block = false;
    }

    return final_records;
}

std::vector<BamPtr> Minimap2Aligner::align_impl(bam1_t* irecord, mm_tbuf_t* buf, int idx_no) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get query name.
    std::string_view qname(bam_get_qname(irecord));

    std::string seq;
    std::string seq_rev;
    std::vector<uint8_t> qual;
    std::vector<uint8_t> qual_rev;

    // If the record is an already aligned record, the strand
    // orientation needs to be fetched so the original
    // read orientation can be recovered.
    bool is_input_reversed = irecord->core.flag & BAM_FREVERSE;

    if (is_input_reversed) {
        seq_rev = utils::extract_sequence(irecord);
        qual_rev = utils::extract_quality(irecord);

        seq = utils::reverse_complement(seq_rev);
        qual = std::vector<uint8_t>(qual_rev.rbegin(), qual_rev.rend());
    } else {
        // get the sequence to map from the record
        seq = utils::extract_sequence(irecord);
        // Pre-generate reverse complement sequence.
        seq_rev = utils::reverse_complement(seq);

        // Pre-generate reverse of quality string.
        qual = utils::extract_quality(irecord);
        qual_rev = std::vector<uint8_t>(qual.rbegin(), qual.rend());
    }

    // do the mapping
    int hits = 0;
    auto mm_index = m_minimap_index->index(idx_no);
    const auto& mm_map_opts = m_minimap_index->mapping_options();
    mm_reg1_t* reg = mm_map(mm_index, static_cast<int>(seq.length()), seq.c_str(), &hits, buf,
                            &mm_map_opts, qname.data());

    // just return the input record
    if (hits == 0) {
        results.push_back(BamPtr(bam_dup1(irecord)));
    }

    for (int j = 0; j < hits; j++) {
        // mapping region
        auto aln = &reg[j];

        // Set FLAGS
        uint16_t flag = 0x0;

        if (aln->rev) {
            flag |= BAM_FREVERSE;
        }
        if (aln->parent != aln->id) {
            flag |= BAM_FSECONDARY;
        } else if (!aln->sam_pri) {
            flag |= BAM_FSUPPLEMENTARY;
        }

        if ((flag & BAM_FSECONDARY) && (mm_map_opts.flag & MM_F_NO_PRINT_2ND)) {
            continue;
        }

        const bool skip_seq_qual = !(mm_map_opts.flag & MM_F_SOFTCLIP) && (flag & BAM_FSECONDARY) &&
                                   !(mm_map_opts.flag & MM_F_SECONDARY_SEQ);
        const bool use_hard_clip =
                !(mm_map_opts.flag & MM_F_SOFTCLIP) &&
                (((flag & BAM_FSECONDARY) && (mm_map_opts.flag & MM_F_SECONDARY_SEQ)) ||
                 (flag & BAM_FSUPPLEMENTARY));
        const auto BAM_CCLIP = use_hard_clip ? BAM_CHARD_CLIP : BAM_CSOFT_CLIP;

        int32_t tid = aln->rid;
        hts_pos_t pos = aln->rs;
        uint8_t mapq = aln->mapq;

        // Create CIGAR.
        // Note: max_bam_cigar_op doesn't need to handled specially when
        // using htslib since the sam_write1 method already takes care
        // of moving the CIGAR string to the tags if the length
        // exceeds 65535.
        size_t n_cigar = aln->p ? aln->p->n_cigar : 0;
        std::vector<uint32_t> cigar;
        uint32_t clip_len[2] = {0};
        if (n_cigar != 0) {
            clip_len[0] = aln->rev ? irecord->core.l_qseq - aln->qe : aln->qs;
            clip_len[1] = aln->rev ? aln->qs : irecord->core.l_qseq - aln->qe;

            if (clip_len[0]) {
                n_cigar++;
            }
            if (clip_len[1]) {
                n_cigar++;
            }
            int offset = clip_len[0] ? 1 : 0;

            cigar.resize(n_cigar);

            // write the left softclip
            if (clip_len[0]) {
                auto clip = bam_cigar_gen(clip_len[0], BAM_CCLIP);
                cigar[0] = clip;
            }

            // write the cigar
            memcpy(&cigar[offset], aln->p->cigar, aln->p->n_cigar * sizeof(uint32_t));

            // write the right softclip
            if (clip_len[1]) {
                auto clip = bam_cigar_gen(clip_len[1], BAM_CCLIP);
                cigar[offset + aln->p->n_cigar] = clip;
            }
        }

        // Add SEQ and QUAL.
        size_t l_seq = 0;
        char* seq_tmp = nullptr;
        unsigned char* qual_tmp = nullptr;
        // To match minimap2 output behavior, don't emit sequence
        // or quality info for secondary alignments.
        if (!skip_seq_qual) {
            l_seq = seq.size();
            if (aln->rev) {
                seq_tmp = seq_rev.data();
                qual_tmp = qual_rev.empty() ? nullptr : qual_rev.data();
            } else {
                seq_tmp = seq.data();
                qual_tmp = qual.empty() ? nullptr : qual.data();
            }
        }
        if (use_hard_clip) {
            l_seq -= clip_len[0] + clip_len[1];
            if (seq_tmp) {
                seq_tmp += clip_len[0];
            }
            if (qual_tmp) {
                qual_tmp += clip_len[0];
            }
        }

        // new output record
        bam1_t* record = bam_init1();

        // Set properties of the BAM record.
        bam_set1(record, qname.size(), qname.data(), flag, tid, pos, mapq, n_cigar,
                 cigar.empty() ? nullptr : cigar.data(), irecord->core.mtid, irecord->core.mpos,
                 irecord->core.isize, l_seq, seq_tmp, (char*)qual_tmp, bam_get_l_aux(irecord));

        // Copy over tags from input alignment.
        memcpy(bam_get_aux(record), bam_get_aux(irecord), bam_get_l_aux(irecord));
        record->l_data += bam_get_l_aux(irecord);

        // Add new tags to match minimap2.
        add_tags(record, aln, seq, buf, idx_no);
        if (!skip_seq_qual) {
            // Here pass the original query length before any hard clip because the
            // the CIGAR string in SA tag only makes use of soft clip. And for that to be
            // correct the unclipped query length is needed.
            add_sa_tag(record, reg, hits, j, static_cast<int>(seq.size()), mm_index);
        }

        // Remove MM/ML/MN tags if secondary alignment and soft clipping is not enabled.
        if ((flag & (BAM_FSUPPLEMENTARY | BAM_FSECONDARY)) && !(mm_map_opts.flag & MM_F_SOFTCLIP)) {
            if (auto tag = bam_aux_get(record, "MM"); tag != nullptr) {
                bam_aux_del(record, tag);
            }
            if (auto tag = bam_aux_get(record, "ML"); tag != nullptr) {
                bam_aux_del(record, tag);
            }
            if (auto tag = bam_aux_get(record, "MN"); tag != nullptr) {
                bam_aux_del(record, tag);
            }
        }

        results.push_back(BamPtr(record));
    }

    // Free all mm2 alignment memory.
    for (int j = 0; j < hits; j++) {
        free(reg[j].p);
    }
    free(reg);
    return results;
}

void Minimap2Aligner::align(dorado::ReadCommon& read_common,
                            const std::string& alignment_header,
                            mm_tbuf_t* buffer) {
    const size_t num_index_blocks = m_minimap_index->num_loaded_index_blocks();
    if (num_index_blocks == 0) {
        throw std::logic_error("Minimap2Aligner::align called without a loaded index.");
    }

    std::vector<std::vector<AlignmentResult>> block_results(num_index_blocks);
    std::vector<int> scores(num_index_blocks);
    bool alignment_found = false;
    for (size_t i = 0; i < num_index_blocks; ++i) {
        auto results = align_impl(read_common, alignment_header, buffer, int(i));
        for (size_t j = 0; j < results.size(); ++j) {
            if (results[j].genome != "*") {
                alignment_found = true;
            }
            if (j == 0) {
                scores[i] = results[j].strand_score;
            }
        }
        block_results[i].swap(results);
    }

    // We need to put the best primary alignment first.
    int best_score = 0;
    size_t best_index = 0;
    for (size_t i = 0; i < num_index_blocks; ++i) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_index = i;
        }
    }
    if (best_index != 0) {
        std::swap(block_results[0], block_results[best_index]);
    }

    // We need to remove any spurious unmapped results.
    size_t non_sup_count = 0;  // Counts non-supplementary records, for mapq score recalculation.
    size_t num_blocks_mapped = 0;  // Counts blocks with mappings, for mapq score recalculation.
    if (alignment_found) {
        bool first_block = true;
        for (auto& results : block_results) {
            std::vector<AlignmentResult> filtered_results;
            // We can skip any unmapped records.
            for (auto& result : results) {
                if (result.genome != "*") {
                    if (!first_block || !result.supplementary_alignment) {
                        non_sup_count++;
                    }
                    filtered_results.emplace_back(std::move(result));
                }
            }
            results.swap(filtered_results);
            first_block = false;
            if (!results.empty()) {
                num_blocks_mapped++;
            }
        }
    } else {
        // We can just include the first block of results, which will only have 1 result.
        block_results.resize(1);
    }

    // Only recompute the mapq score if more than 1 block produced an alignment.
    const int new_mapq = (num_blocks_mapped > 1) ? compute_mapq(non_sup_count) : -1;

    // Mark all alignments that aren't from the first block as secondary.
    bool first_block = true;
    for (auto& results : block_results) {
        for (auto& result : results) {
            update_read_record(result, new_mapq, first_block);
        }
        first_block = false;
    }

    for (auto& results : block_results) {
        for (auto& result : results) {
            read_common.alignment_results.emplace_back(std::move(result));
        }
    }
}

std::vector<AlignmentResult> Minimap2Aligner::align_impl(dorado::ReadCommon& read_common,
                                                         const std::string& alignment_header,
                                                         mm_tbuf_t* buffer,
                                                         int idx_no) {
    mm_bseq1_t query{};
    query.seq = const_cast<char*>(read_common.seq.c_str());
    query.name = const_cast<char*>(read_common.read_id.c_str());
    query.l_seq = static_cast<int>(read_common.seq.length());

    int n_regs{};
    mm_reg1_t* regs = mm_map(m_minimap_index->index(idx_no), query.l_seq, query.seq, &n_regs,
                             buffer, &m_minimap_index->mapping_options(), nullptr);
    auto post_condition = utils::PostCondition([regs, n_regs] {
        for (int reg_idx = 0; reg_idx < n_regs; ++reg_idx) {
            free(regs[reg_idx].p);
        }
        free(regs);
    });

    std::string alignment_string{};
    if (!alignment_header.empty()) {
        alignment_string += alignment_header + "\n";
    }

    if (n_regs == 0) {
        alignment_string = read_common.read_id + UNMAPPED_SAM_LINE_STRIPPED;
    }
    for (int reg_idx = 0; reg_idx < n_regs; ++reg_idx) {
        kstring_t alignment_line{0, 0, nullptr};
        mm_write_sam3(&alignment_line, m_minimap_index->index(idx_no), &query, 0, reg_idx, 1,
                      &n_regs, &regs, NULL, MM_F_OUT_MD, buffer->rep_len);
        alignment_string += std::string(alignment_line.s, alignment_line.l) + "\n";
        free(alignment_line.s);
    }
    return parse_sam_lines(alignment_string, read_common.seq, read_common.qstring);
}

HeaderSequenceRecords Minimap2Aligner::get_sequence_records_for_header() const {
    return m_minimap_index->get_sequence_records_for_header();
}

// Function to add auxiliary tags to the alignment record.
// These are added to maintain parity with mm2.
void Minimap2Aligner::add_tags(bam1_t* record,
                               const mm_reg1_t* aln,
                               const std::string& seq,
                               const mm_tbuf_t* buf,
                               int idx_no) {
    if (aln->p) {
        // NM
        int32_t nm = aln->blen - aln->mlen + aln->p->n_ambi;
        bam_aux_append(record, "NM", 'i', sizeof(nm), (uint8_t*)&nm);

        // ms
        int32_t ms = aln->p->dp_max;
        bam_aux_append(record, "ms", 'i', sizeof(nm), (uint8_t*)&ms);

        // AS
        int32_t as = aln->p->dp_score;
        bam_aux_append(record, "AS", 'i', sizeof(nm), (uint8_t*)&as);

        // nn
        int32_t nn = aln->p->n_ambi;
        bam_aux_append(record, "nn", 'i', sizeof(nm), (uint8_t*)&nn);

        if (aln->p->trans_strand == 1 || aln->p->trans_strand == 2) {
            bam_aux_append(record, "ts", 'A', sizeof(char),
                           (uint8_t*)&("?+-?"[aln->p->trans_strand]));
        }
    }

    // de / dv
    if (aln->p) {
        float div;
        div = static_cast<float>(1.0 - mm_event_identity(aln));
        bam_aux_append(record, "de", 'f', sizeof(div), (uint8_t*)&div);
    } else if (aln->div >= 0.0f && aln->div <= 1.0f) {
        bam_aux_append(record, "dv", 'f', sizeof(aln->div), (uint8_t*)&aln->div);
    }

    // tp
    char type;
    if (aln->id == aln->parent) {
        type = aln->inv ? 'I' : 'P';
    } else {
        type = aln->inv ? 'i' : 'S';
    }
    bam_aux_append(record, "tp", 'A', sizeof(type), (uint8_t*)&type);

    // cm
    bam_aux_append(record, "cm", 'i', sizeof(aln->cnt), (uint8_t*)&aln->cnt);

    // s1
    bam_aux_append(record, "s1", 'i', sizeof(aln->score), (uint8_t*)&aln->score);

    // s2
    if (aln->parent == aln->id) {
        bam_aux_append(record, "s2", 'i', sizeof(aln->subsc), (uint8_t*)&aln->subsc);
    }

    // MD
    char* md = NULL;
    int max_len = 0;
    int md_len = mm_gen_MD(NULL, &md, &max_len, m_minimap_index->index(idx_no), aln, seq.c_str());
    if (md_len > 0) {
        bam_aux_append(record, "MD", 'Z', md_len + 1, (uint8_t*)md);
    }
    free(md);

    // zd
    if (aln->split) {
        uint32_t split = uint32_t(aln->split);
        bam_aux_append(record, "zd", 'i', sizeof(split), (uint8_t*)&split);
    }

    // rl
    bam_aux_append(record, "rl", 'i', sizeof(buf->rep_len), (uint8_t*)&buf->rep_len);
}

}  // namespace dorado::alignment
