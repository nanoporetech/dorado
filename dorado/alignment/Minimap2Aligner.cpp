#include "Minimap2Aligner.h"

#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <minimap.h>
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
}  // namespace

namespace dorado::alignment {

// Stripped of the prefix QNAME and postfix SEQ + \t + QUAL
const std::string UNMAPPED_SAM_LINE_STRIPPED{"\t4\t*\t0\t0\t*\t*\t0\t0\n"};

std::vector<BamPtr> Minimap2Aligner::align(bam1_t* irecord, mm_tbuf_t* buf) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get query name.
    std::string_view qname(bam_get_qname(irecord));

    // get the sequence to map from the record
    std::string seq = utils::extract_sequence(irecord);
    // Pre-generate reverse complement sequence.
    std::string seq_rev = utils::reverse_complement(seq);

    // Pre-generate reverse of quality string.
    std::vector<uint8_t> qual = utils::extract_quality(irecord);
    std::vector<uint8_t> qual_rev(qual.rbegin(), qual.rend());

    // do the mapping
    int hits = 0;
    auto mm_index = m_minimap_index->index();
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
        add_tags(record, aln, seq, buf);
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

void Minimap2Aligner::align(dorado::ReadCommon& read_common, mm_tbuf_t* buffer) {
    mm_bseq1_t query{};
    query.seq = const_cast<char*>(read_common.seq.c_str());
    query.name = const_cast<char*>(read_common.read_id.c_str());
    query.l_seq = static_cast<int>(read_common.seq.length());

    int n_regs{};
    mm_reg1_t* regs = mm_map(m_minimap_index->index(), query.l_seq, query.seq, &n_regs, buffer,
                             &m_minimap_index->mapping_options(), nullptr);
    auto post_condition = utils::PostCondition([regs] { free(regs); });

    std::string alignment_string{};
    if (n_regs == 0) {
        alignment_string = read_common.read_id + UNMAPPED_SAM_LINE_STRIPPED;
    }
    for (int reg_idx{0}; reg_idx < n_regs; ++reg_idx) {
        kstring_t alignment_line{0, 0, nullptr};
        mm_write_sam3(&alignment_line, m_minimap_index->index(), &query, 0, reg_idx, 1, &n_regs,
                      &regs, NULL, MM_F_OUT_MD, buffer->rep_len);
        alignment_string += std::string(alignment_line.s, alignment_line.l) + "\n";
        free(alignment_line.s);
        free(regs[reg_idx].p);
    }
    read_common.alignment_string = alignment_string;
}

HeaderSequenceRecords Minimap2Aligner::get_sequence_records_for_header() const {
    return m_minimap_index->get_sequence_records_for_header();
}

// Function to add auxiliary tags to the alignment record.
// These are added to maintain parity with mm2.
void Minimap2Aligner::add_tags(bam1_t* record,
                               const mm_reg1_t* aln,
                               const std::string& seq,
                               const mm_tbuf_t* buf) {
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
            bam_aux_append(record, "ts", 'A', 2, (uint8_t*)&("?+-?"[aln->p->trans_strand]));
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
    int md_len = mm_gen_MD(NULL, &md, &max_len, m_minimap_index->index(), aln, seq.c_str());
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
