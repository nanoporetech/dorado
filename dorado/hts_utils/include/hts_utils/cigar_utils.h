#pragma once

#include <utility>

struct bam1_t;

namespace dorado::utils {

/**
 * \brief Computes the alignment accuracy from the BAM record.
 *          Requires the NM tag to be present because there is no guarantee that the CIGAR contains =/X ops.
 * \returns pair<accuracy, accuracy_snp> where accuracy is the total alignment accuracy as `nm / aln_len` and
 *              `accuracy_snp` is the accuracy when indels are ignored (`num_x / num_m` where `num_x` is the number
 *              of X CIGAR operations and `num_m` the number of match/mismatch CIGAR operations).
 */
std::pair<double, double> compute_accuracy_from_cigar(const bam1_t *b);

}  // namespace dorado::utils
