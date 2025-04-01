#include "medaka_bamiter.h"

#include <htslib/sam.h>

#include <cerrno>
#include <cstring>

namespace dorado::secondary {

int32_t mpileup_read_bam(void *data, bam1_t *b) {
    HtslibMpileupData *aux = reinterpret_cast<HtslibMpileupData *>(data);

    const bool check_tag = (strcmp(aux->tag_name, "") != 0);
    const bool have_rg = (aux->read_group != nullptr);

    int32_t ret = 0;

    while (true) {
        ret = aux->iter ? sam_itr_next(aux->fp, aux->iter, b) : sam_read1(aux->fp, aux->hdr, b);

        if (ret < 0) {
            break;
        }

        // Skip secondary, supplementary and unmapped alignments.
        if (b->core.flag &
            (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FQCFAIL | BAM_FDUP)) {
            continue;
        }

        // Filter by mapping quality.
        if (static_cast<int32_t>(b->core.qual) < aux->min_mapq) {
            continue;
        }

        // Filter by tag
        if (check_tag) {
            const uint8_t *tag = bam_aux_get(const_cast<const bam1_t *>(b), aux->tag_name);
            if (tag == nullptr) {  // Tag isn't present or is corrupt.
                if (aux->keep_missing) {
                    break;
                } else {
                    continue;
                }
            }
            const int32_t tag_value = static_cast<int32_t>(bam_aux2i(tag));
            if (errno == EINVAL) {
                continue;  // tag was not integer
            }
            if (tag_value != aux->tag_value) {
                continue;
            }
        }

        // Filter by RG (read group).
        if (have_rg) {
            const uint8_t *rg = bam_aux_get((const bam1_t *)b, "RG");
            if (rg == nullptr) {
                continue;  // Missing.
            }
            const char *rg_val = bam_aux2Z(rg);
            if (errno == EINVAL) {
                continue;  // Bad parse.
            }
            if (strcmp(aux->read_group, rg_val) != 0) {
                continue;  // Not wanted.
            }
        }
        break;
    }

    return ret;
}

}  // namespace dorado::secondary
