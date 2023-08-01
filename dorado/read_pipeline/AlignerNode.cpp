#include "AlignerNode.h"

#include "htslib/sam.h"
#include "minimap.h"
//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include "mmpriv.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>
#include <vector>

namespace dorado {

Aligner::Aligner(const std::string& filename, int k, int w, uint64_t index_batch_size, int threads)
        : MessageSink(10000), m_threads(threads) {
    // Check if reference file exists.
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("Aligner reference path does not exist: " + filename);
    }
    // Initialize option structs.
    mm_set_opt(0, &m_idx_opt, &m_map_opt);
    // Setting options to map-ont default till relevant args are exposed.
    mm_set_opt("map-ont", &m_idx_opt, &m_map_opt);

    m_idx_opt.k = k;
    m_idx_opt.w = w;
    spdlog::info("> Index parameters input by user: kmer size={} and window size={}.", m_idx_opt.k,
                 m_idx_opt.w);

    // Set batch sizes large enough to not require chunking since that's
    // not supported yet.
    m_idx_opt.batch_size = index_batch_size;
    m_idx_opt.mini_batch_size = index_batch_size;

    // Force cigar generation.
    m_map_opt.flag |= MM_F_CIGAR;

    mm_check_opt(&m_idx_opt, &m_map_opt);

    m_index_reader = mm_idx_reader_open(filename.c_str(), &m_idx_opt, 0);
    m_index = mm_idx_reader_read(m_index_reader, m_threads);
    auto* split_index = mm_idx_reader_read(m_index_reader, m_threads);
    if (split_index != nullptr) {
        mm_idx_destroy(m_index);
        mm_idx_destroy(split_index);
        mm_idx_reader_close(m_index_reader);
        throw std::runtime_error(
                "Dorado doesn't support split index for alignment. Please re-run with larger index "
                "size.");
    }
    mm_mapopt_update(&m_map_opt, m_index);

    if (m_index->k != m_idx_opt.k || m_index->w != m_idx_opt.w) {
        spdlog::warn(
                "Indexing parameters mismatch prebuilt index: using paramateres kmer "
                "size={} and window size={} from prebuilt index.",
                m_index->k, m_index->w);
    }

    if (mm_verbose >= 3) {
        mm_idx_stat(m_index);
    }

    for (int i = 0; i < m_threads; i++) {
        m_tbufs.push_back(mm_tbuf_init());
    }

    start_threads();
}

void Aligner::start_threads() {
    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&Aligner::worker_thread, this, i)));
    }
}

void Aligner::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
    m_workers.clear();
}

void Aligner::restart() {
    restart_input_queue();
    start_threads();
}

Aligner::~Aligner() {
    terminate_impl();
    for (int i = 0; i < m_threads; i++) {
        mm_tbuf_destroy(m_tbufs[i]);
    }
    mm_idx_reader_close(m_index_reader);
    mm_idx_destroy(m_index);
}

Aligner::bam_header_sq_t Aligner::get_sequence_records_for_header() const {
    std::vector<std::pair<char*, uint32_t>> records;
    for (int i = 0; i < m_index->n_seq; ++i) {
        records.push_back(std::make_pair(m_index->seq[i].name, m_index->seq[i].len));
    }
    return records;
}

void Aligner::worker_thread(size_t tid) {
    Message message;
    while (get_input_message(message)) {
        // If this message isn't a BamPtr, just forward it to the sink.
        if (!std::holds_alternative<BamPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        auto read = std::get<BamPtr>(std::move(message));
        auto records = align(read.get(), m_tbufs[tid]);
        for (auto& record : records) {
            send_message_to_sink(std::move(record));
        }
    }
}

// If an alignment has secondary alignments, add that informatino
// to each record. Follows minimap2 conventions.
void add_sa_tag(bam1_t* record,
                const mm_reg1_t* aln,
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
        bam_aux_append(record, "SA", 'Z', sa.length() + 1, (uint8_t*)sa.c_str());
    }
}

// Function to add auxiliary tags to the alignment record.
// These are added to maintain parity with mm2.
void Aligner::add_tags(bam1_t* record,
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
        div = 1.0 - mm_event_identity(aln);
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
    int md_len = mm_gen_MD(NULL, &md, &max_len, m_index, aln, seq.c_str());
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

std::vector<BamPtr> Aligner::align(bam1_t* irecord, mm_tbuf_t* buf) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;

    // get query name.
    std::string_view qname(bam_get_qname(irecord));

    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
    // Pre-generate reverse complement sequence.
    std::string seq_rev = utils::reverse_complement(seq);

    // Pre-generate reverse of quality string.
    std::vector<uint8_t> qual;
    std::vector<uint8_t> qual_rev;
    if (bam_get_qual(irecord)) {
        qual = std::vector<uint8_t>(bam_get_qual(irecord), bam_get_qual(irecord) + seqlen);
        qual_rev = std::vector<uint8_t>(qual.rbegin(), qual.rend());
    }

    // do the mapping
    int hits = 0;
    mm_reg1_t* reg =
            mm_map(m_index, seq.length(), seq.c_str(), &hits, buf, &m_map_opt, qname.data());

    // just return the input record
    if (hits == 0) {
        results.push_back(BamPtr(bam_dup1(irecord)));
    }

    for (int j = 0; j < hits; j++) {
        // new output record
        bam1_t* record = bam_init1();

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
        if (n_cigar != 0) {
            uint32_t clip_len[2] = {0};
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
                auto clip = bam_cigar_gen(clip_len[0], BAM_CSOFT_CLIP);
                cigar[0] = clip;
            }

            // write the cigar
            memcpy(&cigar[offset], aln->p->cigar, aln->p->n_cigar * sizeof(uint32_t));

            // write the right softclip
            if (clip_len[1]) {
                auto clip = bam_cigar_gen(clip_len[1], BAM_CSOFT_CLIP);
                cigar[offset + aln->p->n_cigar] = clip;
            }
        }

        // Add SEQ and QUAL.
        size_t l_seq = 0;
        char* seq_tmp = nullptr;
        unsigned char* qual_tmp = nullptr;
        if (flag & BAM_FSECONDARY) {
            // To match minimap2 output behavior, don't emit sequence
            // or quality info for secondary alignments.
        } else {
            l_seq = seq.size();
            if (aln->rev) {
                seq_tmp = seq_rev.data();
                qual_tmp = qual_rev.empty() ? nullptr : qual_rev.data();
            } else {
                seq_tmp = seq.data();
                qual_tmp = qual.empty() ? nullptr : qual.data();
            }
        }

        // Set properties of the BAM record.
        // NOTE: Passing bam_get_qname(irecord) + l_qname into bam_set1
        // was causing the generated string to have some extra
        // null characters. Not sure why yet. Using string_view
        // resolved that issue, which is okay to use since it doesn't
        // copy any data and we know the underlying string is null
        // terminated.
        // TODO: See if bam_get_qname(irecord) usage can be fixed.
        bam_set1(record, qname.size(), qname.data(), flag, tid, pos, mapq, n_cigar,
                 cigar.empty() ? nullptr : cigar.data(), irecord->core.mtid, irecord->core.mpos,
                 irecord->core.isize, l_seq, seq_tmp, (char*)qual_tmp, bam_get_l_aux(irecord));

        // Copy over tags from input alignment.
        memcpy(bam_get_aux(record), bam_get_aux(irecord), bam_get_l_aux(irecord));
        record->l_data += bam_get_l_aux(irecord);

        // Add new tags to match minimap2.
        add_tags(record, aln, seq, buf);
        add_sa_tag(record, aln, reg, hits, j, l_seq, m_index);

        results.push_back(BamPtr(record));
    }

    // Free all mm2 alignment memory.
    for (int j = 0; j < hits; j++) {
        free(reg[j].p);
    }
    free(reg);
    return results;
}

stats::NamedStats Aligner::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
