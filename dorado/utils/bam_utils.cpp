#include "bam_utils.h"

#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

std::map<std::string, std::shared_ptr<Read>> read_bam(std::string reads_file) {
    samFile* fp_in = hts_open(reads_file.c_str(), "r");

    bam_hdr_t* bamHdr = sam_hdr_read(fp_in);  //read header
    bam1_t* aln = bam_init1();                //initialize an alignment
    std::cerr << "Header:\n " << bamHdr->text << std::endl;

    std::map<std::string, std::shared_ptr<Read>> reads;

    while (sam_read1(fp_in, bamHdr, aln) >= 0) {
        uint32_t len = aln->core.l_qseq;  //length of the read.

        std::string read_id = bam_get_qname(aln);

        uint8_t* q = bam_get_qual(aln);  //quality string
        uint8_t* s = bam_get_seq(aln);   //sequence string

        std::vector<uint8_t> qualities(len);
        std::vector<char> nucleotides(len);

        // Todo - there is a better way to do this.
        for (int i = 0; i < len; i++) {
            qualities[i] = q[i];
            nucleotides[i] = seq_nt16_str[bam_seqi(s, i)];
        }
        auto tmp_read = std::make_shared<Read>();
        tmp_read->read_id = read_id;
        tmp_read->sequence = nucleotides;
        tmp_read->scores = qualities;
        reads[read_id] = tmp_read;
    }

    bam_destroy1(aln);
    sam_close(fp_in);
    return reads;
}