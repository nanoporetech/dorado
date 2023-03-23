#include "bam_utils.h"

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

    while (reader.next()) {
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

BamReader::~BamReader() {
    if (m_header) {
        bam_hdr_destroy(m_header);
    }
    if (m_record) {
        bam_destroy1(m_record);
    }
    if (m_format) {
        free(m_format);
    }
    if (m_file) {
        sam_close(m_file);
    }
}

bool BamReader::next() { return sam_read1(m_file, m_header, m_record) >= 0; }

BamWriter::BamWriter(const std::string& filename, const sam_hdr_t* header) {
    m_file = hts_open(filename.c_str(), "wb");
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    m_header = sam_hdr_dup(header);
    write_hdr_pg();
    write_hdr_sq();
    auto res = sam_hdr_write(m_file, m_header);
}

BamWriter::~BamWriter() {
    if (m_header) {
        bam_hdr_destroy(m_header);
    }
    if (m_file) {
        sam_close(m_file);
    }
}

int BamWriter::write_record(bam1_t* record) { return sam_write1(m_file, m_header, record); }

int BamWriter::write_record(bam1_t* record,
                            uint16_t flag,
                            int32_t tid,
                            hts_pos_t pos,
                            uint8_t mapq) {
    int n_cigar = 1;
    uint32_t ssize = record->core.l_qseq;
    uint32_t cigar[] = {ssize << BAM_CIGAR_SHIFT | BAM_CMATCH};

    record->core.flag = flag;
    record->core.tid = tid;
    record->core.pos = pos;
    record->core.qual = mapq;
    record->core.n_cigar = n_cigar;

    if (n_cigar != 0) {
        int cigar_size = n_cigar * sizeof(uint32_t);
        uint8_t* data = (uint8_t*)realloc(record->data, record->l_data + cigar_size);
        record->data = data;

        // Shift existing data to make room for the new cigar field
        memmove(record->data + record->core.l_qname + cigar_size,
                record->data + record->core.l_qname, record->l_data - record->core.l_qname);

        // Copy the new cigar field into the bam1_t structure
        memcpy(record->data + record->core.l_qname, cigar, cigar_size);

        // Update the data length
        record->l_data += cigar_size;
    }

    return sam_write1(m_file, m_header, record);
}

int BamWriter::write_hdr_pg() {
    return sam_hdr_add_line(m_header, "PG", "ID", "aligner", "PN", "dorado", "VN", MM_VERSION, "DS",
                            "minimap2", NULL);  // add CL Writer node
}

int BamWriter::write_hdr_sq() {
    return sam_hdr_add_line(m_header, "SQ", "SN", "CHM1", "LN", "1000", NULL);
}

}  // namespace dorado::utils
