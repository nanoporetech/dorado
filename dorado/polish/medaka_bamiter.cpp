#include "medaka_bamiter.h"

#include <errno.h>
#include <string.h>

// iterator for reading bam
int read_bam(void *data, bam1_t *b) {
    mplp_data *aux = (mplp_data *)data;
    uint8_t *tag;
    bool check_tag = (strcmp(aux->tag_name, "") != 0);
    bool have_rg = (aux->read_group != NULL);
    uint8_t *rg;
    char *rg_val;
    int ret;
    while (1) {
        ret = aux->iter ? sam_itr_next(aux->fp, aux->iter, b) : sam_read1(aux->fp, aux->hdr, b);
        if (ret < 0) {
            break;
        }
        // only take primary alignments
        if (b->core.flag &
            (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FQCFAIL | BAM_FDUP)) {
            continue;
        }
        // filter by mapping quality
        if ((int)b->core.qual < aux->min_mapQ) {
            continue;
        }
        // filter by tag
        if (check_tag) {
            tag = bam_aux_get((const bam1_t *)b, aux->tag_name);
            if (tag == NULL) {  // tag isn't present or is currupt
                if (aux->keep_missing) {
                    break;
                } else {
                    continue;
                }
            }
            int tag_value = static_cast<int32_t>(bam_aux2i(tag));
            if (errno == EINVAL) {
                continue;  // tag was not integer
            }
            if (tag_value != aux->tag_value) {
                continue;
            }
        }
        // filter by RG (read group):
        if (have_rg) {
            rg = bam_aux_get((const bam1_t *)b, "RG");
            if (rg == NULL) {
                continue;  // missing
            }
            rg_val = bam_aux2Z(rg);
            if (errno == EINVAL) {
                continue;  // bad parse
            }
            if (strcmp(aux->read_group, rg_val) != 0) {
                continue;  // not wanted
            }
        }
        break;
    }
    return ret;
}

// Initialise BAM file, index and header structures
bam_fset *create_bam_fset(const char *fname) {
    bam_fset *fset = (bam_fset *)calloc(1, sizeof(bam_fset));
    if (fset == NULL) {
        fprintf(stderr, "Failed to allocate mem for bam fileset.\n");
        exit(1);
    }

    fset->fp = hts_open(fname, "rb");
    fset->idx = sam_index_load(fset->fp, fname);
    fset->hdr = sam_hdr_read(fset->fp);
    if (fset->hdr == 0 || fset->idx == 0 || fset->fp == 0) {
        destroy_bam_fset(fset);
        fprintf(stderr, "Failed to read .bam file '%s'.", fname);
        exit(1);
    }
    return fset;
}

// Destory BAM file, index and header structures
void destroy_bam_fset(bam_fset *fset) {
    hts_close(fset->fp);
    hts_idx_destroy(fset->idx);
    sam_hdr_destroy(fset->hdr);
    free(fset);
}
