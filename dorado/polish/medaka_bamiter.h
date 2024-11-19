#pragma once

#include "htslib/sam.h"

#include <stdbool.h>

// parameters for bam iteration
typedef struct {
    htsFile *fp;
    sam_hdr_t *hdr;
    hts_itr_t *iter;
    int min_mapQ;
    char tag_name[2];
    int tag_value;
    bool keep_missing;
    const char *read_group;
} mplp_data;

typedef struct {
    htsFile *fp;
    hts_idx_t *idx;
    sam_hdr_t *hdr;
} bam_fset;

// Initialise BAM file, index and header structures
bam_fset *create_bam_fset(const char *fname);

// Destory BAM file, index and header structures
void destroy_bam_fset(bam_fset *fset);

// iterator for reading bam
int read_bam(void *data, bam1_t *b);
