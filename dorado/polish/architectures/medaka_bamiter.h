#pragma once

#include <htslib/sam.h>

struct HtslibMpileupData {
    htsFile* fp;
    sam_hdr_t* hdr;
    hts_itr_t* iter;
    int min_mapQ;
    char tag_name[2];
    int tag_value;
    bool keep_missing;
    const char* read_group;
};

// iterator for reading bam
int read_bam(void* data, bam1_t* b);
