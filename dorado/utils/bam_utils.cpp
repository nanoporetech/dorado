#include "bam_utils.h"

#include "Version.h"
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

std::map<std::string, std::shared_ptr<Read>> read_bam(const std::string& filename,
                                                      const std::set<std::string>& read_ids) {
    BamReader reader(filename);
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

Aligner::Aligner(const std::string& filename, const int threads) {
    m_threads = threads;

    mm_set_opt(0, &m_idx_opt, &m_map_opt);

    m_idx_opt.k = 19;
    m_idx_opt.w = 19;
    m_idx_opt.flag = 1;
    m_idx_opt.bucket_bits = 14;
    m_idx_opt.batch_size = 16000000000;

    m_map_opt.flag |= MM_F_CIGAR;

    m_index_reader = mm_idx_reader_open(filename.c_str(), &m_idx_opt, 0);
    m_index = mm_idx_reader_read(m_index_reader, m_threads);
    mm_mapopt_update(&m_map_opt, m_index);

    if (mm_verbose >= 3) {
        mm_idx_stat(m_index);
    }

    for (int i = 0; i < m_threads; i++) {
        m_tbufs.push_back(mm_tbuf_init());
    }
}

Aligner::~Aligner() {
    mm_idx_destroy(m_index);
    for (int i = 0; i < m_threads; i++) {
        mm_tbuf_destroy(m_tbufs[i]);
    }
    mm_idx_reader_close(m_index_reader);
}

std::pair<int, mm_reg1_t*> Aligner::align(const std::vector<char> seq) {
    int hits = 0;
    mm_reg1_t* reg = mm_map(m_index, seq.size(), seq.data(), &hits, m_tbufs[0], &m_map_opt, 0);
    return std::make_pair(hits, reg);
}

std::vector<bam1_t*> Aligner::align(bam1_t* irecord) {
    // some where for the hits
    std::vector<bam1_t*> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;

    auto bseq = bam_get_seq(irecord);
    std::vector<char> nucleotides(seqlen);
    for (int i = 0; i < seqlen; i++) {
        nucleotides[i] = seq_nt16_str[bam_seqi(bseq, i)];
    }

    // do the mapping
    int hits = 0;
    mm_reg1_t* reg = mm_map(m_index, nucleotides.size(), nucleotides.data(), &hits, m_tbufs[0],
                            &m_map_opt, 0);

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

        // todo: handle max_bam_cigar_op

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
            uint8_t* data = (uint8_t*)realloc(record->data, record->l_data + cigar_size);

            // Shift existing data to make room for the new cigar field
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

            // Update the data length
            record->l_data += cigar_size;

            //TODO: m_data size
            //TODO: extra tags (NM)
        }
        free(a->p);
        results.push_back(record);
    }

    free(reg);
    return results;
}

std::vector<std::pair<char*, uint32_t>> Aligner::get_idx_records() {
    std::vector<std::pair<char*, uint32_t>> records;
    for (int i = 0; i < m_index->n_seq; ++i) {
        records.push_back(std::make_pair(m_index->seq[i].name, m_index->seq[i].len));
    }
    return records;
}

BamReader::BamReader(const std::string& filename) {
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

bool BamReader::read() { return sam_read1(m_file, m_header, m_record) >= 0; }

BamReader::~BamReader() {
    free(m_format);
    sam_hdr_destroy(m_header);
    bam_destroy1(m_record);
    hts_close(m_file);
}

BamWriter::BamWriter(const std::string& filename, const sam_hdr_t* header, sq_t seqs) {
    m_file = hts_open(filename.c_str(), "wb");
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    m_header = sam_hdr_dup(header);
    write_hdr_pg();

    for (auto pair : seqs) {
        write_hdr_sq(std::get<0>(pair), std::get<1>(pair));
    }
    auto res = sam_hdr_write(m_file, m_header);
}

BamWriter::~BamWriter() {
    sam_hdr_destroy(m_header);
    hts_close(m_file);
}

int BamWriter::write(bam1_t* record) { return sam_write1(m_file, m_header, record); }

int BamWriter::write_hdr_pg() {
    //TODO: add CL Writer node
    return sam_hdr_add_line(m_header, "PG", "ID", "aligner", "PN", "dorado", "VN", DORADO_VERSION,
                            "DS", MM_VERSION, NULL);
}

int BamWriter::write_hdr_sq(char* name, uint32_t length) {
    return sam_hdr_add_line(m_header, "SQ", "SN", name, "LN", std::to_string(length).c_str(), NULL);
}

}  // namespace dorado::utils
