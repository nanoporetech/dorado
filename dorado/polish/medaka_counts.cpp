#include "medaka_counts.h"

#include "htslib/sam.h"
#include "medaka_bamiter.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <unordered_set>

#define bam1_seq(b) ((b)->data + (b)->core.n_cigar * 4 + (b)->core.l_qname)
#define bam1_seqi(s, i) (bam_seqi((s), (i)))

namespace dorado::polisher {

PileupData::PileupData(const int64_t n_cols,
                       const int64_t buffer_cols,
                       const int64_t num_dtypes,
                       const int64_t num_homop,
                       const int64_t fixed_size)
        : m_buffer_cols{buffer_cols},
          m_num_dtypes{num_dtypes},
          m_num_homop{num_homop},
          m_n_cols{n_cols} {
    if (fixed_size > 0) {
        assert(buffer_cols == n_cols);
        m_matrix.resize(fixed_size * n_cols, 0);
    } else {
        m_matrix.resize(PILEUP_BASES_SIZE * num_dtypes * buffer_cols * num_homop, 0);
    }
    m_major.resize(buffer_cols);
    m_minor.resize(buffer_cols);
}

void PileupData::resize_cols(const int64_t buffer_cols) {
    const int64_t new_size = PILEUP_BASES_SIZE * m_num_dtypes * m_num_homop * buffer_cols;
    m_matrix.resize(new_size, 0);
    m_major.resize(buffer_cols, 0);
    m_minor.resize(buffer_cols, 0);
    m_buffer_cols = buffer_cols;
}

/** Prints a pileup data structure.
 *
 *  @param pileup a pileup structure.
 *  @param num_dtypes number of datatypes in the pileup.
 *  @param dtypes datatype prefix strings.
 *  @param num_homop maximum homopolymer length to consider.
 *  @returns void
 *
 */
void print_pileup_data(std::ostream &os,
                       const PileupData &pileup,
                       const int64_t num_dtypes,
                       const std::vector<std::string> &dtypes,
                       const int64_t num_homop) {
    os << "pos\tins\t";
    if (num_dtypes > 1) {  //TODO header for multiple dtypes and num_homop > 1
        for (int64_t i = 0; i < num_dtypes; ++i) {
            for (int64_t j = 0; j < PILEUP_BASES_SIZE; ++j) {
                os << dtypes[i] << '.' << PILEUP_BASES[j] << '\t';
            }
        }
    } else {
        for (int64_t k = 0; k < num_homop; ++k) {
            for (int64_t j = 0; j < PILEUP_BASES_SIZE; ++j) {
                os << PILEUP_BASES[j] << '.' << (k + 1) << '\t';
            }
        }
    }
    os << "depth\n";
    for (int64_t j = 0; j < static_cast<int64_t>(pileup.n_cols()); ++j) {
        int64_t s = 0;
        os << pileup.get_major()[j] << '\t' << pileup.get_minor()[j] << '\t';
        for (int64_t i = 0; i < static_cast<int64_t>(num_dtypes * PILEUP_BASES_SIZE * num_homop);
             ++i) {
            const int64_t c =
                    pileup.get_matrix().at(j * num_dtypes * PILEUP_BASES_SIZE * num_homop + i);
            s += c;
            os << c << '\t';
        }
        os << s << '\n';
    }
}

std::vector<float> _get_weibull_scores(const bam_pileup1_t *p,
                                       const int64_t indel,
                                       const int64_t num_homop,
                                       std::unordered_set<std::string> &bad_reads) {
    // Create homopolymer scores using Weibull shape and scale parameters.
    // If prerequisite sam tags are not present an array of zero counts is returned.
    std::vector<float> fraction_counts(num_homop);
    static const char *wtags[] = {"WL", "WK"};  // scale, shape
    double wtag_vals[2] = {0.0, 0.0};
    for (int64_t i = 0; i < 2; ++i) {
        uint8_t *tag = bam_aux_get(p->b, wtags[i]);
        if (tag == NULL) {
            const std::string read_id(bam_get_qname(p->b));
            const auto it = bad_reads.find(read_id);
            if (it == std::end(bad_reads)) {  // a new bad read
                bad_reads.emplace(read_id);
                spdlog::warn("Failed to retrieve Weibull parameter tag '{}' for read {}.\n",
                             wtags[i], read_id);
            }
            return fraction_counts;
        }
        uint32_t taglen = bam_auxB_len(tag);
        if (p->qpos + indel >= taglen) {
            spdlog::warn("%s tag was out of range for %s position %lu. taglen: %i\n", wtags[i],
                         bam_get_qname(p->b), p->qpos + indel, taglen);
            return fraction_counts;
        }
        wtag_vals[i] = bam_auxB2f(tag, p->qpos + static_cast<int32_t>(indel));
    }

    // found tags, fill in values
    float scale = static_cast<float>(wtag_vals[0]);  //wl
    float shape = static_cast<float>(wtag_vals[1]);  //wk
    for (int64_t x = 1; x < num_homop + 1; ++x) {
        float a = std::pow((x - 1) / scale, shape);
        float b = std::pow(x / scale, shape);
        fraction_counts[x - 1] = std::fmax(0.0f, -std::exp(-a) * std::expm1(a - b));
    }
    return fraction_counts;
}

/** Generates medaka-style feature data in a region of a bam.
 *
 *  @param region 1-based region string.
 *  @param bam_file input aligment file.
 *  @param num_dtypes number of datatypes in bam.
 *  @param dtypes prefixes on query names indicating datatype.
 *  @param num_homop maximum homopolymer length to consider.
 *  @param tag_name by which to filter alignments.
 *  @param tag_value by which to filter data.
 *  @param keep_missing alignments which do not have tag.
 *  @param weibull_summation use predefined bam tags to perform homopolymer partial counts.
 *  @returns a pileup data pointer.
 *
 *  The return value can be freed with destroy_plp_data.
 *
 *  If num_dtypes is 1, dtypes should be NULL; all reads in the bam will be
 *  treated equally. If num_dtypes is not 1, dtypes should be an array of
 *  strings, these strings being prefixes of query names of reads within the
 *  bam file. Any read not matching the prefixes will cause exit(1).
 *
 *  If tag_name is not NULL alignments are filtered by the (integer) tag value.
 *  When tag_name is given the behaviour for alignments without the tag is
 *  determined by keep_missing.
 *
 */
PileupData calculate_pileup(BamFile &bam_file,
                            const std::string &chr_name,
                            const int64_t start,  // Zero-based.
                            const int64_t end,    // Non-inclusive.

                            const int64_t num_dtypes,
                            const std::vector<std::string> &dtypes,
                            const int64_t num_homop,
                            const std::string &tag_name,
                            const int32_t tag_value,
                            const bool keep_missing,
                            const bool weibull_summation,
                            const char *read_group,
                            const int32_t min_mapq) {
    if ((num_dtypes == 1) && !std::empty(dtypes)) {
        throw std::runtime_error(
                "Recieved invalid num_dtypes and dtypes args. num_dtypes == 1 but size(dtypes) = " +
                std::to_string(std::size(dtypes)));
    }
    if (num_dtypes == 0) {
        throw std::runtime_error("The num_dtypes needs to be > 0.");
    }

    const int64_t dtype_featlen = PILEUP_BASES_SIZE * num_dtypes * num_homop;

    // open bam etc.
    // this is all now deferred to the caller
    htsFile *fp = bam_file.fp();
    hts_idx_t *idx = bam_file.idx();
    sam_hdr_t *hdr = bam_file.hdr();

    const std::string region =
            chr_name + ':' + std::to_string(start + 1) + '-' + std::to_string(end);

    std::unique_ptr<HtslibMpileupData> data = std::make_unique<HtslibMpileupData>();
    HtslibMpileupData *raw_data_ptr = data.get();

    data->fp = fp;
    data->hdr = hdr;
    data->iter = bam_itr_querys(idx, hdr, region.c_str());
    data->min_mapQ = min_mapq;
    memcpy(data->tag_name, tag_name.c_str(), 2);
    data->tag_value = tag_value;
    data->keep_missing = keep_missing;
    data->read_group = read_group;
    bam_mplp_t mplp = bam_mplp_init(1, read_bam, reinterpret_cast<void **>(&raw_data_ptr));

    std::array<bam_pileup1_t *, 1> plp;
    // const bam_pileup1_t** plp_ptr = &plp.data();

    int32_t ret = 0;
    int32_t pos = 0;
    int32_t tid = 0;
    int32_t n_plp = 0;

    // allocate output assuming one insertion per ref position
    int32_t n_cols = 0;
    const int64_t buffer_cols = 2 * (end - start);
    PileupData pileup(n_cols, buffer_cols, num_dtypes, num_homop, 0);

    int64_t *pileup_matrix = pileup.get_matrix().data();
    int64_t *pileup_major = pileup.get_major().data();
    int64_t *pileup_minor = pileup.get_minor().data();

    // get counts
    int64_t major_col = 0;  // index into `pileup` corresponding to pos
    n_cols = 0;             // number of processed columns (including insertions)
    std::unordered_set<std::string> no_rle_tags;

    while ((ret = bam_mplp_auto(mplp, &tid, &pos, &n_plp,
                                const_cast<const bam_pileup1_t **>(plp.data()))) > 0) {
        const char *c_name = data->hdr->target_name[tid];
        if (c_name != chr_name) {
            continue;
        }
        if (pos < start) {
            continue;
        }
        if (pos >= end) {
            break;
        }
        n_cols++;

        // find maximum insert
        int32_t max_ins = 0;
        for (int32_t i = 0; i < n_plp; ++i) {
            const bam_pileup1_t *p = plp[0] + i;
            if ((p->indel > 0) && (max_ins < p->indel)) {
                max_ins = p->indel;
            }
        }

        // reallocate output if necessary
        if ((n_cols + max_ins) > static_cast<int32_t>(pileup.buffer_cols())) {
            const float cols_per_pos = static_cast<float>(n_cols + max_ins) / (1 + pos - start);
            // max_ins can dominate so add at least that
            const int64_t new_buffer_cols =
                    max_ins + std::max(2 * pileup.buffer_cols(),
                                       static_cast<int64_t>(cols_per_pos * (end - start)));
            pileup.resize_cols(new_buffer_cols);
            pileup_matrix = pileup.get_matrix().data();
            pileup_major = pileup.get_major().data();
            pileup_minor = pileup.get_minor().data();
        }

        // set major/minor position indexes, minors hold ins
        for (int32_t i = 0; i <= max_ins; ++i) {
            pileup_major[major_col / dtype_featlen + i] = pos;
            pileup_minor[major_col / dtype_featlen + i] = i;
        }

        // loop through all reads at this position
        for (int32_t i = 0; i < n_plp; ++i) {
            const bam_pileup1_t *p = plp[0] + i;
            if (p->is_refskip) {
                continue;
            }

            // find to which datatype the read belongs
            int32_t dtype = 0;
            if (num_dtypes > 1) {
                bool failed = false;
                char *tag_val;
                uint8_t *tag = bam_aux_get(p->b, DATATYPE_TAG.data());
                if (tag == NULL) {  // tag isn't present
                    failed = true;
                } else {
                    tag_val = bam_aux2Z(tag);
                    failed = errno == EINVAL;
                }
                if (!failed) {
                    bool found = false;
                    for (dtype = 0; dtype < static_cast<int32_t>(num_dtypes); ++dtype) {
                        if (strcmp(dtypes[dtype].c_str(), tag_val) == 0) {
                            found = true;
                            break;
                        }
                    }
                    failed = !found;
                }
                if (failed) {
                    throw std::runtime_error("Datatype not found for '" +
                                             std::string(bam_get_qname(p->b)) + "'.");
                }
            }

            int32_t base_i = 0;
            int32_t min_minor = 0;
            const int32_t max_minor = (p->indel > 0) ? p->indel : 0;
            if (p->is_del) {
                // deletions are kept in the first layer of qscore stratification, if any
                int32_t qstrat = 0;
                base_i = bam_is_rev(p->b) ? PILEUP_POS_DEL_REV : PILEUP_POS_DEL_FWD;
                pileup_matrix[major_col + PILEUP_BASES_SIZE * dtype * num_homop +
                              PILEUP_BASES_SIZE * qstrat + base_i] += 1;
                min_minor = 1;  // in case there is also an indel, skip the major position
            }
            // loop over any query bases at or inserted after pos
            int32_t query_pos_offset = 0;
            for (int32_t minor = min_minor; minor <= max_minor; ++minor, ++query_pos_offset) {
                int32_t base_j = bam1_seqi(bam1_seq(p->b), p->qpos + query_pos_offset);
                //base = seq_nt16_str[base_j];
                if bam_is_rev (p->b) {
                    base_j += 16;
                }

                base_i = NUM_TO_COUNT_BASE[base_j];
                if (base_i != -1) {  // not an ambiguity code
                    const int64_t partial_index =
                            major_col + dtype_featlen * minor        // skip to column
                            + PILEUP_BASES_SIZE * dtype * num_homop  // skip to datatype
                            //+ PILEUP_BASES_SIZE * qstrat                           // skip to qstrat/homop
                            + base_i;  // the base

                    if (weibull_summation) {
                        const std::vector<float> fraction_counts =
                                _get_weibull_scores(p, query_pos_offset, num_homop, no_rle_tags);
                        for (int64_t qstrat = 0; qstrat < num_homop; ++qstrat) {
                            static const int32_t scale = 10000;
                            pileup_matrix[partial_index + PILEUP_BASES_SIZE * qstrat] +=
                                    static_cast<int64_t>(scale * fraction_counts[qstrat]);
                        }
                    } else {
                        int32_t qstrat = 0;
                        if (num_homop > 1) {
                            // want something in [0, num_homop-1]
                            qstrat = static_cast<int32_t>(std::min<int64_t>(
                                    bam_get_qual(p->b)[p->qpos + query_pos_offset], num_homop));
                            qstrat = std::max(0, qstrat - 1);
                        }
                        pileup_matrix[partial_index + PILEUP_BASES_SIZE * qstrat] += 1;
                    }
                }
            }
        }
        major_col += (dtype_featlen * (max_ins + 1));
        n_cols += max_ins;
    }

    pileup.n_cols(n_cols);
    pileup.get_major().resize(n_cols);
    pileup.get_minor().resize(n_cols);

    bam_itr_destroy(data->iter);
    bam_mplp_destroy(mplp);

    return pileup;
}

}  // namespace dorado::polisher
