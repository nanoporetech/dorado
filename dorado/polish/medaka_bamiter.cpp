#include "medaka_bamiter.h"

#include <cerrno>
#include <cstring>

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