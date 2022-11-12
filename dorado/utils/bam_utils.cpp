#include "bam_utils.h"

#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
// seq_nt16_str is referred to in the hts-3.lib stub on windows, but has not been declared dllimport for
//  client code, so it comes up as an undefined reference when linking the stub.
const char seq_nt16_str[] = "=ACMGRSVTWYHKDBN";
#endif  // _WIN32

namespace dorado::utils {
std::map<std::string, std::shared_ptr<Read>> read_bam(std::string reads_file) {
    samFile* fp_in = hts_open(reads_file.c_str(), "r");

    bam_hdr_t* bamHdr = sam_hdr_read(fp_in);  //read header
    bam1_t* aln = bam_init1();                //initialize an alignment

    std::map<std::string, std::shared_ptr<Read>> reads;

    while (sam_read1(fp_in, bamHdr, aln) >= 0) {
        uint32_t seqlen = aln->core.l_qseq;

        std::string read_id = bam_get_qname(aln);

        uint8_t* qstring = bam_get_qual(aln);
        uint8_t* sequence = bam_get_seq(aln);

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

    bam_destroy1(aln);
    sam_close(fp_in);
    return reads;
}
}  // namespace dorado::utils
