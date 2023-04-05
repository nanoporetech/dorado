#include "bam_utils.h"

#include "Version.h"
#include "htslib/kroundup.h"
#include "htslib/sam.h"
#include "minimap.h"
#include "read_pipeline/ReadPipeline.h"

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
// seq_nt16_str is referred to in the hts-3.lib stub on windows, but has not been declared dllimport for
//  client code, so it comes up as an undefined reference when linking the stub.
const char seq_nt16_str[] = "=ACMGRSVTWYHKDBN";
#endif  // _WIN32

namespace dorado::utils {

Aligner::Aligner(MessageSink& sink, const std::string& filename, const int threads)
        : MessageSink(10000), m_sink(sink), m_threads(threads) {
    mm_set_opt(0, &m_idx_opt, &m_map_opt);

    m_idx_opt.k = 19;
    m_idx_opt.w = 19;
    m_idx_opt.flag = 1;
    m_idx_opt.batch_size = 4000000000;
    m_idx_opt.mini_batch_size = 16000000000;

    m_map_opt.flag |= MM_F_CIGAR;

    mm_check_opt(&m_idx_opt, &m_map_opt);

    m_index_reader = mm_idx_reader_open(filename.c_str(), &m_idx_opt, 0);
    m_index = mm_idx_reader_read(m_index_reader, m_threads);
    mm_mapopt_update(&m_map_opt, m_index);

    if (mm_verbose >= 3) {
        mm_idx_stat(m_index);
    }

    for (int i = 0; i < m_threads; i++) {
        m_tbufs.push_back(mm_tbuf_init());
    }

    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&Aligner::worker_thread, this)));
    }
}

Aligner::~Aligner() {
    for (auto& m : m_workers) {
        m->join();
    }
    for (int i = 0; i < m_threads; i++) {
        mm_tbuf_destroy(m_tbufs[i]);
    }
    mm_idx_reader_close(m_index_reader);
    mm_idx_destroy(m_index);
}

std::vector<std::pair<char*, uint32_t>> Aligner::sq() {
    std::vector<std::pair<char*, uint32_t>> records;
    for (int i = 0; i < m_index->n_seq; ++i) {
        records.push_back(std::make_pair(m_index->seq[i].name, m_index->seq[i].len));
    }
    return records;
}

std::pair<int, mm_reg1_t*> Aligner::align(const std::vector<char> seq) {
    int hits = 0;
    mm_reg1_t* reg = mm_map(m_index, seq.size(), seq.data(), &hits, m_tbufs[0], &m_map_opt, 0);
    return std::make_pair(hits, reg);
}

void Aligner::worker_thread() {
    Message message;
    int tid = m_active++ % m_threads;
    while (m_work_queue.try_pop(message)) {
        auto records = align(std::get<bam1_t*>(message), m_tbufs[tid]);
        for (auto& record : records) {
            m_sink.push_message(std::move(record));
        }
    }
    if (--m_active == 0) {
        terminate();
        m_sink.terminate();
    }
}

// Function to add auxiliary tags to the alignment record.
// These are added to maintain parity with mm2.
void Aligner::add_tags(bam1_t* record, const mm_reg1_t* aln, const std::vector<char>& seq) {
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

        if (aln->p->trans_strand == 1 || aln->p->trans_strand == 2)
            bam_aux_append(record, "ts", 'A', 2, (uint8_t*)&("?+-?"[aln->p->trans_strand]));
    }

    // tp
    char type;
    if (aln->id == aln->parent)
        type = aln->inv ? 'I' : 'P';
    else
        type = aln->inv ? 'i' : 'S';
    bam_aux_append(record, "tp", 'A', sizeof(type), (uint8_t*)&type);

    // cm
    bam_aux_append(record, "cm", 'i', sizeof(aln->cnt), (uint8_t*)&aln->cnt);

    // s1
    bam_aux_append(record, "s1", 'i', sizeof(aln->score), (uint8_t*)&aln->score);

    // s2
    if (aln->parent == aln->id)
        bam_aux_append(record, "s2", 'i', sizeof(aln->subsc), (uint8_t*)&aln->subsc);

    // MD
    char* md = NULL;
    int max_len = 0;
    int md_len = mm_gen_MD(NULL, &md, &max_len, m_index, aln, seq.data());
    if (md_len > 0)
        bam_aux_append(record, "MD", 'Z', md_len + 1, (uint8_t*)md);

    // zd
    if (aln->split) {
        uint32_t split = uint32_t(aln->split);
        bam_aux_append(record, "zd", 'i', sizeof(split), (uint8_t*)&split);
    }

    // TODO: do we need the de and dv tags? They use an mm_event_identity function
    // which is private in mm2, so we will have to duplicate that functionality if we need these
    // tags.
}

std::vector<bam1_t*> Aligner::align(bam1_t* irecord, mm_tbuf_t* buf) {
    // some where for the hits
    std::vector<bam1_t*> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;

    auto bseq = bam_get_seq(irecord);
    std::vector<char> seq(seqlen);
    for (int i = 0; i < seqlen; i++) {
        seq[i] = seq_nt16_str[bam_seqi(bseq, i)];
    }

    // do the mapping
    int hits = 0;
    mm_reg1_t* reg = mm_map(m_index, seq.size(), seq.data(), &hits, buf, &m_map_opt, 0);

    // just return the input record
    if (hits == 0) {
        results.push_back(irecord);
    }

    for (int j = 0; j < hits; j++) {
        // new output record
        auto record = bam_dup1(irecord);

        // mapping region
        auto a = &reg[j];

        uint16_t flag = 0x0;

        if (a->rev)
            flag |= 0x10;
        if (a->parent != a->id)
            flag |= 0x100;
        else if (!a->sam_pri)
            flag |= 0x800;

        record->core.flag = flag;
        record->core.tid = a->rid;
        record->core.pos = a->rs;
        record->core.qual = a->mapq;
        record->core.n_cigar = a->p ? a->p->n_cigar : 0;

        // Note: max_bam_cigar_op doesn't need to handled specially when
        // using htslib since the sam_write1 method already takes care
        // of moving the CIGAR string to the tags if the length
        // exceeds 65535.
        if (record->core.n_cigar != 0) {
            uint32_t clip_len[2] = {0};
            clip_len[0] = a->rev ? record->core.l_qseq - a->qe : a->qs;
            clip_len[1] = a->rev ? a->qs : record->core.l_qseq - a->qe;

            if (clip_len[0]) {
                record->core.n_cigar++;
            }
            if (clip_len[1]) {
                record->core.n_cigar++;
            }
            int offset = clip_len[0] ? 1 : 0;
            int cigar_size = record->core.n_cigar * sizeof(uint32_t);
            uint32_t new_m_data = record->l_data + cigar_size;
            kroundup32(new_m_data);
            uint8_t* data = (uint8_t*)realloc(record->data, new_m_data);

            // shift existing data to make room for the new cigar field
            memmove(data + record->core.l_qname + cigar_size, data + record->core.l_qname,
                    record->l_data - record->core.l_qname);

            record->data = data;

            // write the left softclip
            if (clip_len[0]) {
                auto clip = bam_cigar_gen(clip_len[0], BAM_CSOFT_CLIP);
                memcpy(bam_get_cigar(record), &clip, sizeof(uint32_t));
            }

            // write the cigar
            memcpy(bam_get_cigar(record) + offset, a->p->cigar, a->p->n_cigar * sizeof(uint32_t));

            // write the right softclip
            if (clip_len[1]) {
                auto clip = bam_cigar_gen(clip_len[1], BAM_CSOFT_CLIP);
                memcpy(bam_get_cigar(record) + offset + a->p->n_cigar, &clip, sizeof(uint32_t));
            }

            // update the data length
            record->l_data += cigar_size;
            record->m_data = new_m_data;
        }

        add_tags(record, a, seq);

        free(a->p);
        results.push_back(record);
    }

    free(reg);
    return results;
}

BamReader::BamReader(MessageSink& sink, const std::string& filename) : m_sink(sink) {
    m_file = hts_open(filename.c_str(), "r");
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    m_format = hts_format_description(hts_get_format(m_file));
    m_header = sam_hdr_read(m_file);
    if (!m_header) {
        throw std::runtime_error("Could not read header from file: " + filename);
    }
    m_is_aligned = m_header->n_targets > 0;
    m_record = bam_init1();
}

BamReader::~BamReader() {
    free(m_format);
    sam_hdr_destroy(m_header);
    bam_destroy1(m_record);
    hts_close(m_file);
}

void BamReader::read(int max_reads) {
    int num_reads = 0;
    while (sam_read1(m_file, m_header, m_record) >= 0) {
        m_sink.push_message(std::move(bam_dup1(m_record)));
        if (++num_reads >= max_reads) {
            break;
        }
    }
    m_sink.terminate();
}

BamWriter::BamWriter(const std::string& filename) : MessageSink(1000) {
    m_file = hts_open(filename.c_str(), "wb");
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    m_worker = std::make_unique<std::thread>(std::thread(&BamWriter::worker_thread, this));
}

BamWriter::~BamWriter() {
    sam_hdr_destroy(m_header);
    hts_close(m_file);
}

void BamWriter::join() { m_worker->join(); }

void BamWriter::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        write(std::get<bam1_t*>(message));
    }
    terminate();
}

int BamWriter::write(bam1_t* record) {
    // track stats
    m_total++;
    if (record->core.flag & BAM_FUNMAP)
        m_unmapped++;
    if (record->core.flag & BAM_FSECONDARY)
        m_secondary++;
    if (record->core.flag & BAM_FSUPPLEMENTARY)
        m_supplementary++;
    m_primary = m_total - m_secondary - m_supplementary - m_unmapped;

    auto res = sam_write1(m_file, m_header, record);
    if (res < 0) {
        throw std::runtime_error("Failed to write SAM record, error code " + std::to_string(res));
    }
    return res;
}

int BamWriter::write_header(const sam_hdr_t* header, const sq_t seqs) {
    m_header = sam_hdr_dup(header);
    write_hdr_pg();
    for (auto pair : seqs) {
        write_hdr_sq(std::get<0>(pair), std::get<1>(pair));
    }
    auto res = sam_hdr_write(m_file, m_header);
    return res;
}

int BamWriter::write_hdr_pg() {
    // todo: add CL Writer node
    return sam_hdr_add_line(m_header, "PG", "ID", "aligner", "PN", "dorado", "VN", DORADO_VERSION,
                            "DS", MM_VERSION, NULL);
}

int BamWriter::write_hdr_sq(char* name, uint32_t length) {
    return sam_hdr_add_line(m_header, "SQ", "SN", name, "LN", std::to_string(length).c_str(), NULL);
}

/*
read_map read_bam(const std::string& filename, const std::set<std::string>& read_ids) {

    utils::sq_t seq;
    sam_hdr_t* hdr;

    utils::BamWriter writer("-", hdr, seq);

    BamReader reader(writer, filename);

    std::map<std::string, std::shared_ptr<Read>> reads;

    while (reader.read()) {
        std::string read_id = bam_get_qname(reader.m_record);

        if (read_ids.find(read_id) == read_ids.end()) {
            continue;
        }

        uint8_t* qstring = bam_get_qual(reader.m_record);
        uint8_t* sequence = bam_get_seq(reader.m_record);

        uint32_t seqlen = reader.m_record->core.l_qseq;
        std::vector<uint8_t> qualities(seqlen);
        std::vector<char> nucleotides(seqlen);

        // Todo - there is a better way to do this.
        for (int i = 0; i < seqlen; i++) {
            qualities[i] = qstring[i] + 33;
            nucleotides[i] = seq_nt16_str[bam_seqi(sequence, i)];
        }

        auto tmp_read = std::make_shared<Read>();
        tmp_read->read_id = read_id;
        tmp_read->seq = std::string(nucleotides.begin(), nucleotides.end());
        tmp_read->qstring = std::string(qualities.begin(), qualities.end());
        reads[read_id] = tmp_read;
    }

    return reads;
}
*/

}  // namespace dorado::utils
