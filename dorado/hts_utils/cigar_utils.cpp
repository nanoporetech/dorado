#include "hts_utils/cigar_utils.h"

#include <htslib/sam.h>

#include <algorithm>
#include <cstdint>

namespace dorado::utils {

std::pair<double, double> compute_accuracy_from_cigar(const bam1_t *b) {
    if (!b) {
        return {0.0, 0.0};
    }

    // Retrieve NM tag (edit distance)
    const uint8_t *nm_tag = bam_aux_get(b, "NM");
    if (!nm_tag) {
        // NOTE: Not reporting a warning because some test datasets may not contain NM tags.
        // spdlog::warn("NM tag not found for read: {}", bam_get_qname(b));
        return {0.0, 0.0};
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
        // const CigarOpType op = CIGAR_MM2_TO_DORADO[op_int];

        switch (op_int) {
        case BAM_CMATCH:
        case BAM_CEQUAL:
        case BAM_CDIFF:
            num_m += op_len;
            break;
        case BAM_CINS:
            num_ins += op_len;
            break;
        case BAM_CDEL:
            num_del += op_len;
            break;
        default:
            continue;
        }
    }

    const int64_t aligned_query_len = num_m + num_ins;
    const int64_t alignment_len = num_m + num_ins + num_del;

    if ((aligned_query_len <= 0) || (alignment_len <= 0) || (num_m <= 0)) {
        return {0.0, 0.0};
    }

    const int64_t num_x = nm - num_ins - num_del;

    const double acc_x = (num_m == 0) ? 0.0 : std::clamp((1.0 - ((double)num_x) / num_m), 0.0, 1.0);
    const double acc_total =
            (alignment_len == 0) ? 0.0 : std::clamp((1.0 - ((double)nm) / alignment_len), 0.0, 1.0);

    return {acc_total, acc_x};
}

}  // namespace dorado::utils
