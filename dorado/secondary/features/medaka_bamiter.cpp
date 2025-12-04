#include "medaka_bamiter.h"

#include "utils/cigar.h"

#include <htslib/sam.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace dorado::secondary {

namespace {
double compute_accuracy_from_cigar(const bam1_t *b, const bool mismatch_only) {
    if (!b) {
        return 0.0;
    }

    // Retrieve NM tag (edit distance)
    const uint8_t *nm_tag = bam_aux_get(b, "NM");
    if (!nm_tag) {
        return 0.0;
    }
    const int64_t nm = bam_aux2i(nm_tag);

    // Compute aligned query length from CIGAR
    const uint32_t *cigar = bam_get_cigar(b);
    int64_t num_ins = 0;
    int64_t num_del = 0;
    int64_t num_m = 0;

    for (uint32_t i = 0; i < b->core.n_cigar; ++i) {
        const int op_int = bam_cigar_op(cigar[i]);
        const int op_len = bam_cigar_oplen(cigar[i]);
        const CigarOpType op = CIGAR_MM2_TO_DORADO[op_int];

        switch (op) {
        case CigarOpType::M:
        case CigarOpType::EQ:
        case CigarOpType::X:
            num_m += op_len;
            break;
        case CigarOpType::I:
            num_ins += op_len;
            break;
        case CigarOpType::D:
            num_del += op_len;
            break;
        default:
            continue;
        }
    }

    const int64_t alignment_len = num_m + num_ins + num_del;

    if ((num_m <= 0) || (alignment_len <= 0)) {
        return 0.0;
    }

    // Filter on SNP accuracy.
    if (mismatch_only) {
        const int64_t num_x = nm - num_ins - num_del;
        return std::clamp(1.0 - static_cast<double>(num_x) / static_cast<double>(num_m), 0.0, 1.0);
    }

    // Classic alignment accuracy filtering.
    return std::clamp(1.0 - static_cast<double>(nm) / static_cast<double>(alignment_len), 0.0, 1.0);
}
}  // namespace

int32_t mpileup_read_bam(void *data, bam1_t *b) {
    HtslibMpileupData *aux = reinterpret_cast<HtslibMpileupData *>(data);

    const bool check_tag = (strcmp(aux->tag_name, "") != 0);
    const bool have_rg = (aux->read_group != nullptr);
    const double min_snp_accuracy = aux->min_snp_accuracy;

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

        // Filter by accuracy if available.
        const double accuracy = compute_accuracy_from_cigar(b, true);
        if ((accuracy >= 0.0) && (accuracy < min_snp_accuracy)) {
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
