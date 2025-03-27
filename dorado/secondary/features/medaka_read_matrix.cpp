#include "medaka_read_matrix.h"

#include "medaka_bamiter.h"
#include "secondary/bam_file.h"
#include "utils/ssize.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#define bam1_seq(b) ((b)->data + (b)->core.n_cigar * 4 + (b)->core.l_qname)
#define bam1_seqi(s, i) (bam_seqi((s), (i)))

namespace {

static constexpr int32_t BASE_FEATLEN = 4;  // Minimal number of feature channels.
static constexpr int8_t DEL_VAL = 5;        // Value representing deletion in base channel.
static constexpr std::string_view DATATYPE_TAG{"DT", 2};

// convert 16bit IUPAC (+16 for strand) to plp_bases index
static constexpr std::array<int8_t, 32> NUM_TO_COUNT_BASE_SYMM{
        -1, 1, 2, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1,
        -1, 1, 2, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1,
};

struct Read {
    int64_t ref_start;
    std::string qname;
    uint8_t *seqi;
    uint8_t *qual;
    uint8_t mapq;
    int8_t haplotype;
    int8_t strand;
    int8_t dtype;
    int64_t ref_end;
    std::vector<int8_t> dwells;
};

/** Populate an array of dwells per base.
 *
 *  \param alignment an htslib alignment.
 *  \param ret_dwells return vector of dwells.
 *  \returns status of dwell computation (success, no dwell tags or bad alignment).
 */
enum class CalcDwellsReturnValue {
    SUCCESS,
    NO_DWELL_TAG,
    BAD_ALIGNMENT,
};
CalcDwellsReturnValue calculate_dwells(const bam1_t *alignment, std::vector<int8_t> &ret_dwells) {
    ret_dwells.clear();

    if (alignment == nullptr) {
        return CalcDwellsReturnValue::BAD_ALIGNMENT;
    }

    const int32_t length = alignment->core.l_qseq;
    constexpr int32_t MAX_INT8 = std::numeric_limits<int8_t>::max();

    ret_dwells.clear();
    ret_dwells.resize(length, 0);

    const uint8_t *mv_tag = bam_aux_get(alignment, "mv");
    if (!mv_tag) {
        // Return, but don't clear the ret_dwells so that an empty column is added.
        return CalcDwellsReturnValue::NO_DWELL_TAG;
    }

    const int32_t mv_len = static_cast<int32_t>(bam_auxB_len(mv_tag));

    int64_t qpos = 0;  // Base index.

    if (alignment->core.flag & BAM_FREVERSE) {
        int32_t dwell = 0;
        // Reversed alignment, iterate backward through move table.
        // Last entry is the first move which corresponds to
        // the last base.
        for (int32_t i = (mv_len - 1); i > 0; --i) {
            ++dwell;
            if (bam_auxB2i(mv_tag, i) == 1) {
                if (qpos >= length) {
                    // Return and clear data, this alignment is all wrong.
                    ret_dwells.clear();
                    return CalcDwellsReturnValue::BAD_ALIGNMENT;
                }
                ret_dwells[qpos] = static_cast<int8_t>(std::min(dwell, MAX_INT8));
                ++qpos;
                dwell = 0;
            }
        }
    } else {
        int32_t dwell = 1;
        // Skip first entry since this is always a move.
        // Last entry is the last sample point so need to
        // add the dwell since the last move afterwards.
        for (int32_t i = 2; i < mv_len; ++i) {
            if (bam_auxB2i(mv_tag, i) == 1) {
                if (qpos >= length) {
                    // Return and clear data, this alignment is all wrong.
                    ret_dwells.clear();
                    return CalcDwellsReturnValue::BAD_ALIGNMENT;
                }
                ret_dwells[qpos] = static_cast<int8_t>(std::min(dwell, MAX_INT8));
                ++qpos;
                dwell = 0;
            }
            ++dwell;
        }

        if (qpos >= length) {
            // Return and clear data, this alignment is all wrong.
            ret_dwells.clear();
            return CalcDwellsReturnValue::BAD_ALIGNMENT;
        }

        ret_dwells[qpos] = static_cast<int8_t>(std::min(dwell, MAX_INT8));
    }

    return CalcDwellsReturnValue::SUCCESS;
}

size_t aligned_ref_pos_from_cigar(const uint32_t *cigar, const uint32_t n_cigar) {
    uint32_t aligned_ref_pos = 0;
    for (size_t ci = 0; ci < n_cigar; ++ci) {
        const uint32_t cigar_len = cigar[ci] >> 4;
        const uint8_t cigar_op = cigar[ci] & 0xf;
        if ((cigar_op == BAM_CMATCH) || (cigar_op == BAM_CDEL) || (cigar_op == BAM_CEQUAL) ||
            (cigar_op == BAM_CDIFF)) {
            aligned_ref_pos += cigar_len;
        }
    }
    return aligned_ref_pos;
}

}  // namespace

namespace dorado::secondary {

/** Constructs a ReadAlignmentData data structure.
 *
 *  \param n_pos_ Number of pileup positions (columns).
 *  \param n_reads_ Number of pileup reads (rows).
 *  \param buffer_pos_ Number of pileup positions.
 *  \param buffer_reads_ Number of pileup reads.
 *  \param extra_featlen_ Number of extra feature channels.
 *  \param fixed_size_ If not zero data matrix is allocated as fixed_size * n_reads * n_pos, ignoring other arguments
 */
ReadAlignmentData::ReadAlignmentData(const int32_t n_pos_,
                                     const int32_t n_reads_,
                                     const int32_t buffer_pos_,
                                     const int32_t buffer_reads_,
                                     const int32_t extra_featlen_,
                                     const int32_t fixed_size_)
        : buffer_pos{buffer_pos_},
          buffer_reads{buffer_reads_},
          num_dtypes{1},
          n_pos{n_pos_},
          n_reads{n_reads_},
          featlen{BASE_FEATLEN + extra_featlen_} {
    if (fixed_size_ > 0) {
        assert(buffer_pos == n_pos);
        matrix.resize(fixed_size_ * n_reads * n_pos, 0);
    } else {
        matrix.resize(featlen * buffer_reads * buffer_pos, 0);
    }
    major.resize(buffer_pos);
    minor.resize(buffer_pos);
    read_ids_left.resize(buffer_reads);
    read_ids_right.resize(buffer_reads);
}

void ReadAlignmentData::resize_cols(const int32_t new_buffer_cols) {
    const int64_t new_size = static_cast<int64_t>(new_buffer_cols) * buffer_reads * featlen;
    matrix.resize(new_size, 0);
    major.resize(new_buffer_cols, 0);
    minor.resize(new_buffer_cols, 0);
    buffer_pos = new_buffer_cols;
}

void ReadAlignmentData::resize_num_reads(const int32_t new_buffer_reads) {
    const int64_t old_size = static_cast<int64_t>(buffer_pos) * this->buffer_reads * featlen;
    const int64_t new_size = static_cast<int64_t>(buffer_pos) * new_buffer_reads * featlen;

    matrix.resize(new_size, 0);
    read_ids_left.resize(new_buffer_reads);
    read_ids_right.resize(new_buffer_reads);

    // move old data to the new part of matrix
    for (int64_t p = (buffer_pos - 1); p > 0; --p) {
        // Precompute coords in the outer loop.
        const int64_t old_coord_part = p * this->buffer_reads * featlen;
        const int64_t new_coord_part = p * new_buffer_reads * featlen;
        for (int64_t r = (this->buffer_reads - 1); r >= 0; --r) {
            // Precompute coords in the outer loop.
            const int64_t old_coord_part_2 = old_coord_part + r * featlen;
            const int64_t new_coord_part_2 = new_coord_part + r * featlen;
            for (int64_t f = (featlen - 1); f >= 0; --f) {
                const int64_t old_coord = old_coord_part_2 + f;
                const int64_t new_coord = new_coord_part_2 + f;
                matrix[new_coord] = matrix[old_coord];
            }
        }
    }

    // Zero out old entries.
    for (int64_t p = 0; p < buffer_pos; ++p) {
        // Precompute coords in the outer loop.
        const int64_t old_coord_1 = p * new_buffer_reads * featlen;
        for (int64_t r = this->buffer_reads; r < new_buffer_reads; ++r) {
            // Precompute coords in the outer loop.
            const int64_t old_coord_2 = old_coord_1 + r * featlen;
            for (int64_t f = 0; f < featlen; ++f) {
                const int64_t old_coord = old_coord_2 + f;
                if (old_coord < old_size) {
                    matrix[old_coord] = 0;
                }
            }
        }
    }

    buffer_reads = new_buffer_reads;
}

ReadAlignmentData calculate_read_alignment(secondary::BamFile &bam_file,
                                           const std::string &chr_name,
                                           const int64_t start,
                                           const int64_t end,  // Non-inclusive.
                                           const int64_t num_dtypes,
                                           const std::vector<std::string> &dtypes,
                                           const std::string &tag_name,
                                           const int32_t tag_value,
                                           const bool keep_missing,
                                           const std::string &read_group,
                                           const int32_t min_mapq,
                                           const bool row_per_read,
                                           const bool include_dwells,
                                           const bool include_haplotype,
                                           const int32_t max_reads) {
    if ((num_dtypes == 1) && !std::empty(dtypes)) {
        throw std::runtime_error(
                "Received invalid num_dtypes and dtypes args. num_dtypes == 1 but size(dtypes) = " +
                std::to_string(std::size(dtypes)));
    }
    if (num_dtypes == 0) {
        throw std::runtime_error("The num_dtypes needs to be > 0.");
    }

    // Open bam etc. This is all now deferred to the caller
    htsFile *fp = bam_file.fp();
    hts_idx_t *idx = bam_file.idx();
    sam_hdr_t *hdr = bam_file.hdr();

    if (!fp || !idx || !hdr) {
        throw std::runtime_error{"[calculate_read_alignment] BamFile not opened properly!"};
    }

    const std::string region =
            chr_name + ':' + std::to_string(start + 1) + '-' + std::to_string(end);

    std::unique_ptr<HtslibMpileupData> data = std::make_unique<HtslibMpileupData>();
    HtslibMpileupData *raw_data_ptr = data.get();
    data->fp = fp;
    data->hdr = hdr;
    data->iter = bam_itr_querys(idx, hdr, region.c_str());
    data->min_mapq = min_mapq;
    memcpy(data->tag_name, tag_name.c_str(), 2);
    data->tag_value = tag_value;
    data->keep_missing = keep_missing;
    data->read_group = std::empty(read_group) ? nullptr : read_group.c_str();
    bam_mplp_t mplp = bam_mplp_init(1, mpileup_read_bam, reinterpret_cast<void **>(&raw_data_ptr));

    std::array<bam_pileup1_t *, 1> plp;
    int32_t ret = 0;
    int32_t pos = 0;
    int32_t tid = 0;
    int32_t n_plp = 0;

    // allocate output assuming one insertion per ref position
    int32_t n_pos = 0;
    int32_t max_n_reads = 0;
    const int32_t buffer_pos = static_cast<int32_t>(2 * (end - start));
    const int32_t buffer_reads = std::min(max_reads, 100);
    const int32_t extra_featlen =
            (include_dwells ? 1 : 0) + (include_haplotype ? 1 : 0) + (num_dtypes > 1 ? 1 : 0);
    ReadAlignmentData pileup(n_pos, max_n_reads, buffer_pos, buffer_reads, extra_featlen, 0);

    int64_t major_col = 0;  // index into `pileup` corresponding to pos
    n_pos = 0;              // number of processed columns (including insertions)
    int64_t min_gap = 5;    // minimum gap before starting a new read on an existing row

    // A vector to store all read structs.
    std::vector<Read> read_array;

    // hash map from read ids to index in above vector.
    std::unordered_map<std::string, int32_t> read_map;

    while ((ret = bam_mplp_auto(mplp, &tid, &pos, &n_plp,
                                const_cast<const bam_pileup1_t **>(std::data(plp)))) > 0) {
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
        ++n_pos;

        // find maximum insert and number of reads
        int32_t max_ins = 0;
        for (int32_t i = 0; i < n_plp; ++i) {
            const bam_pileup1_t *p = plp[0] + i;
            if ((p->indel > 0) && (max_ins < p->indel)) {
                max_ins = p->indel;
            }
        }
        max_n_reads = std::max(max_n_reads, n_plp);

        // Reallocate output if necessary.
        if ((n_pos + max_ins) > pileup.buffer_pos) {
            const float cols_per_pos = static_cast<float>(n_pos + max_ins) / (1 + pos - start);
            // max_ins can dominate so add at least that.
            const int32_t new_buffer_pos =
                    max_ins + std::max(2 * pileup.buffer_pos,
                                       static_cast<int32_t>(cols_per_pos * (end - start)));
            pileup.resize_cols(new_buffer_pos);
        }
        const int32_t row_offset = row_per_read ? n_plp : 0;
        if ((pileup.buffer_reads < max_reads) &&
            ((max_n_reads + row_offset) > pileup.buffer_reads)) {
            const int32_t new_buffer_reads = std::min(
                    max_reads, std::max(max_n_reads + row_offset, 2 * pileup.buffer_reads));
            // Correct the start position of the column we're about to write.
            major_col = (major_col / pileup.buffer_reads) * new_buffer_reads;
            pileup.resize_num_reads(new_buffer_reads);
        }
        // Set major/minor position indexes, minors hold ins.
        const int64_t col_idx = major_col / (pileup.featlen * pileup.buffer_reads);
        if ((col_idx < 0) || ((col_idx + max_ins) >= dorado::ssize(pileup.major))) {
            throw std::runtime_error{
                    "[calculate_read_alignment] Index out of bounds: col_idx = " +
                    std::to_string(col_idx) + ", max_ins = " + std::to_string(max_ins) +
                    ", pileup.major.size = " + std::to_string(std::size(pileup.major))};
        }
        for (int i = 0; i <= max_ins; ++i) {
            pileup.major[col_idx + i] = pos;
            pileup.minor[col_idx + i] = i;
        }

        // loop through all reads at this position
        for (int32_t i = 0; i < n_plp; ++i) {
            const bam_pileup1_t *p = plp[0] + i;
            if (p == nullptr) {
                throw std::runtime_error{"[calculate_read_alignment] p is nullptr! i = " +
                                         std::to_string(i)};
            }
            if (p->is_refskip) {
                continue;
            }

            const bam1_t *alignment = p->b;
            if (alignment == nullptr) {
                throw std::runtime_error{"[calculate_read_alignment] Alignment is nullptr!"};
            }

            const std::string qname(bam_get_qname(alignment));

            // check whether read is in hash list
            const auto it = read_map.find(qname);
            int32_t read_i = (it == std::end(read_map)) ? -1 : it->second;
            if (read_i < 0) {
                // get dtype tag
                int32_t dtype = 0;
                bool failed = false;
                if (num_dtypes > 1) {
                    char *tag_val = nullptr;
                    const uint8_t *tag = bam_aux_get(alignment, std::data(DATATYPE_TAG));
                    if (tag == NULL) {  // tag isn't present
                        failed = true;
                    } else {
                        tag_val = bam_aux2Z(tag);
                        failed = errno == EINVAL;
                    }
                    if (!failed) {
                        bool found = false;
                        for (dtype = 0; dtype < num_dtypes; ++dtype) {
                            if (tag_val && (dtypes[dtype] == tag_val)) {
                                found = true;
                                break;
                            }
                        }
                        failed = !found;
                    }
                    if (failed) {
                        throw std::runtime_error{"Datatype not found for qname: '" + qname + "'."};
                    }
                }
                // get haplotype tag
                int8_t haplotype = 0;
                failed = false;
                const uint8_t *tag = bam_aux_get(alignment, "HP");
                if (tag == NULL) {  // tag isn't present
                    failed = true;
                } else {
                    haplotype = static_cast<int8_t>(bam_aux2i(tag));
                    failed = errno == EINVAL;
                }

                std::vector<int8_t> dwells;
                if (include_dwells) {
                    const CalcDwellsReturnValue rv = calculate_dwells(alignment, dwells);
                    if ((rv != CalcDwellsReturnValue::SUCCESS) &&
                        (rv != CalcDwellsReturnValue::NO_DWELL_TAG)) {
                        throw std::runtime_error{"Bad BAM alignment for qname: '" + qname +
                                                 "', could not extract tags!"};
                    }
                }

                Read read = {
                        alignment->core.pos,
                        qname,
                        bam_get_seq(alignment),
                        bam_get_qual(alignment),
                        alignment->core.qual,
                        haplotype,
                        static_cast<int8_t>(bam_is_rev(alignment) ? -1 : 1),
                        static_cast<int8_t>(dtype),
                        static_cast<int64_t>(alignment->core.pos +
                                             aligned_ref_pos_from_cigar(bam_get_cigar(alignment),
                                                                        alignment->core.n_cigar)),
                        std::move(dwells),
                };

                // insert read into read_array in place of a read that's already completed
                const int32_t array_size = static_cast<int32_t>(std::size(read_array));
                if (!row_per_read) {
                    for (read_i = 0; read_i < array_size; ++read_i) {
                        const Read &current_read = read_array[read_i];
                        if (pos >= (current_read.ref_end + min_gap)) {
                            read_array[read_i] = std::move(read);
                            read_map[qname] = read_i;
                            break;
                        }
                    }
                } else {
                    read_i = array_size;
                    if (array_size > max_n_reads) {
                        max_n_reads = array_size;
                    }
                }
                // no completed reads, append instead
                if (read_i == array_size) {
                    if (read_i < pileup.buffer_reads) {
                        read_array.emplace_back(std::move(read));
                    }
                    read_map[qname] = read_i;
                }
            }

            if ((read_i < 0) || (read_i >= (pileup.buffer_reads))) {
                continue;
            }

            const Read &read = read_array[read_i];
            if (n_pos == 1) {
                pileup.read_ids_left[read_i] = read.qname;
            }

            int32_t min_minor = 0;
            const int32_t max_minor = (p->indel > 0) ? p->indel : 0;
            if (p->is_del) {
                pileup.matrix[major_col + pileup.featlen * read_i + 0] = DEL_VAL;      //base
                pileup.matrix[major_col + pileup.featlen * read_i + 1] = -1;           //qual
                pileup.matrix[major_col + pileup.featlen * read_i + 2] = read.strand;  //strand
                pileup.matrix[major_col + pileup.featlen * read_i + 3] = read.mapq;    //mapq
                if (include_dwells) {
                    pileup.matrix[major_col + pileup.featlen * read_i + BASE_FEATLEN] = -1;  //dwell
                }
                if (include_haplotype) {
                    pileup.matrix[major_col + pileup.featlen * read_i + BASE_FEATLEN +
                                  (include_dwells ? 1 : 0)] = read.haplotype;  //haplotag
                }
                if (num_dtypes > 1) {
                    pileup.matrix[major_col + pileup.featlen * read_i + BASE_FEATLEN +
                                  (include_dwells ? 1 : 0) + (include_haplotype ? 1 : 0)] =
                            read.dtype;  //dtype
                }
                min_minor = 1;  // in case there is also an indel, skip the major position
            }
            // loop over any query bases at or inserted after pos
            int32_t query_pos_offset = 0;
            int32_t minor = min_minor;
            for (; minor <= max_minor; ++minor, ++query_pos_offset) {
                const int32_t base_j = bam1_seqi(read.seqi, p->qpos + query_pos_offset);
                const int8_t base_i = NUM_TO_COUNT_BASE_SYMM[base_j];
                const size_t partial_index =
                        major_col + pileup.featlen * pileup.buffer_reads * minor  // skip to column
                        + pileup.featlen * read_i;  // skip to read row

                pileup.matrix[partial_index + 0] = base_i;                                 //base
                pileup.matrix[partial_index + 1] = read.qual[p->qpos + query_pos_offset];  //qual
                pileup.matrix[partial_index + 2] = read.strand;                            //strand
                pileup.matrix[partial_index + 3] = read.mapq;                              //qual
                if (include_dwells && !std::empty(read.dwells)) {
                    pileup.matrix[partial_index + BASE_FEATLEN] =
                            read.dwells[p->qpos + query_pos_offset];  //dwell
                }
                if (include_haplotype) {
                    pileup.matrix[partial_index + BASE_FEATLEN + (include_dwells ? 1 : 0)] =
                            read.haplotype;  //haplotag
                }
                if (num_dtypes > 1) {
                    pileup.matrix[partial_index + BASE_FEATLEN + (include_dwells ? 1 : 0) +
                                  (include_haplotype ? 1 : 0)] = read.dtype;  //dtype
                }
            }
            for (; minor <= max_ins; ++minor) {
                const size_t partial_index =
                        major_col + pileup.featlen * pileup.buffer_reads * minor  // skip to column
                        + pileup.featlen * read_i;  // skip to read row

                pileup.matrix[partial_index + 0] = DEL_VAL;      //base
                pileup.matrix[partial_index + 1] = -1;           //qual
                pileup.matrix[partial_index + 2] = read.strand;  //strand
                pileup.matrix[partial_index + 3] = read.mapq;    //qual
                if (include_dwells) {
                    pileup.matrix[partial_index + BASE_FEATLEN] = -1;  //dwell
                }
                if (include_haplotype) {
                    pileup.matrix[partial_index + BASE_FEATLEN + (include_dwells ? 1 : 0)] =
                            read.haplotype;  //haplotag
                }
                if (num_dtypes > 1) {
                    pileup.matrix[partial_index + BASE_FEATLEN + (include_dwells ? 1 : 0) +
                                  (include_haplotype ? 1 : 0)] = read.dtype;  //dtype
                }
            }
        }
        major_col += (pileup.featlen * pileup.buffer_reads * (max_ins + 1));
        n_pos += max_ins;
    }

    for (size_t r = 0, nleft = 0, nright = 0; r < std::size(read_array); ++r) {
        const Read &read = read_array[r];
        if (read.ref_end >= pos) {
            pileup.read_ids_right[r] = read.qname;
        } else {
            ++nright;
            pileup.read_ids_right[r] = "__blank_" + std::to_string(nright);
        }
        if (std::empty(pileup.read_ids_left[r])) {
            ++nleft;
            pileup.read_ids_left[r] = "__blank_" + std::to_string(nleft);
        }
    }

    pileup.n_pos = n_pos;
    if (row_per_read) {
        pileup.n_reads = static_cast<int32_t>(std::size(read_array));
    } else {
        pileup.n_reads = max_n_reads;
    }
    pileup.n_reads = std::min(max_reads, pileup.n_reads);

    pileup.major.resize(n_pos);
    pileup.minor.resize(n_pos);
    pileup.read_ids_left.resize(pileup.n_reads);
    pileup.read_ids_right.resize(pileup.n_reads);

    bam_itr_destroy(data->iter);
    bam_mplp_destroy(mplp);

    return pileup;
}

}  // namespace dorado::secondary
