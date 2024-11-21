#define _GNU_SOURCE
#include "htslib/khash_str2int.h"
#include "htslib/sam.h"
#include "kvec.h"
#include "medaka_bamiter.h"
#include "medaka_common.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "medaka_counts.h"
#include "medaka_read_matrix.h"

#define bam1_seq(b) ((b)->data + (b)->core.n_cigar * 4 + (b)->core.l_qname)
#define bam1_seqi(s, i) (bam_seqi((s), (i)))

/** Constructs a pileup data structure.
 *
 *  @param n_pos number of pileup positions (columns).
 *  @param n_reads number of pileup reads (rows).
 *  @param buffer_pos number of pileup positions.
 *  @param buffer_reads number of pileup reads.
 *  @param extra_featlen number of extra feature channels.
 *  @param fixed_size if not zero data matrix is allocated as fixed_size * n_reads * n_pos, ignoring other arguments
 *  @see destroy_read_aln_data
 *  @returns a read_aln_data pointer.
 *
 *  The return value can be freed with destroy_read_aln_data.
 *
 */
read_aln_data create_read_aln_data(size_t n_pos,
                                   size_t n_reads,
                                   size_t buffer_pos,
                                   size_t buffer_reads,
                                   size_t extra_featlen,
                                   size_t fixed_size) {
    assert(buffer_pos >= n_pos);
    assert(buffer_reads >= n_reads);
    read_aln_data data = xalloc(1, sizeof(_read_aln_data), "read_aln_data");
    data->buffer_pos = buffer_pos;
    data->buffer_reads = buffer_reads;
    data->featlen = base_featlen + extra_featlen;
    data->n_pos = n_pos;
    data->n_reads = n_reads;
    if (fixed_size != 0) {
        assert(buffer_pos == n_pos);
        data->matrix = xalloc(fixed_size * n_reads * n_pos, sizeof(int8_t), "matrix");
    } else {
        data->matrix = xalloc(data->featlen * buffer_reads * buffer_pos, sizeof(int8_t), "matrix");
    }
    data->major = xalloc(buffer_pos, sizeof(size_t), "major");
    data->minor = xalloc(buffer_pos, sizeof(size_t), "minor");
    data->read_ids_left = xalloc(buffer_reads, sizeof(char *), "read_ids_left");
    data->read_ids_right = xalloc(buffer_reads, sizeof(char *), "read_ids_right");
    return data;
}

/** Enlarge the internal buffers of a pileup data structure.
 *
 *  @param pileup a read_aln_data pointer.
 *  @param buffer_pos new number of pileup positions for which to allocate memory
 *
 */
void enlarge_read_aln_data_pos(read_aln_data pileup, size_t buffer_pos) {
    assert(buffer_pos > pileup->buffer_pos);
    size_t old_size = pileup->buffer_pos * pileup->buffer_reads * pileup->featlen;
    size_t new_size = buffer_pos * pileup->buffer_reads * pileup->featlen;

    pileup->matrix = xrealloc(pileup->matrix, new_size * sizeof(int8_t), "matrix");
    pileup->major = xrealloc(pileup->major, buffer_pos * sizeof(size_t), "major");
    pileup->minor = xrealloc(pileup->minor, buffer_pos * sizeof(size_t), "minor");
    // zero out new part of matrix
    for (size_t i = old_size; i < new_size; ++i) {
        pileup->matrix[i] = 0;
    }
    pileup->buffer_pos = buffer_pos;
}

void enlarge_read_aln_data_reads(read_aln_data pileup, size_t buffer_reads) {
    assert(buffer_reads > pileup->buffer_reads);
    size_t old_size = pileup->buffer_pos * pileup->buffer_reads * pileup->featlen;
    size_t new_size = pileup->buffer_pos * buffer_reads * pileup->featlen;

    pileup->matrix = xrealloc(pileup->matrix, new_size * sizeof(int8_t), "matrix");
    pileup->read_ids_left =
            xrealloc(pileup->read_ids_left, buffer_reads * sizeof(char *), "read_ids_left");
    pileup->read_ids_right =
            xrealloc(pileup->read_ids_right, buffer_reads * sizeof(char *), "read_ids_right");
    for (size_t i = pileup->buffer_reads; i < buffer_reads; ++i) {
        pileup->read_ids_left[i] = NULL;
        pileup->read_ids_right[i] = NULL;
    }
    // move old data to the new part of matrix
    for (size_t p = pileup->buffer_pos - 1; p > 0; --p) {
        for (size_t r = pileup->buffer_reads; r > 0; --r) {
            for (size_t f = pileup->featlen; f > 0; --f) {
                size_t old_coord = p * pileup->buffer_reads * pileup->featlen +
                                   (r - 1) * pileup->featlen + (f - 1);
                size_t new_coord =
                        p * buffer_reads * pileup->featlen + (r - 1) * pileup->featlen + (f - 1);
                pileup->matrix[new_coord] = pileup->matrix[old_coord];
            }
        }
    }
    // zero out old entries
    for (size_t p = 0; p < pileup->buffer_pos; ++p) {
        for (size_t r = pileup->buffer_reads; r < buffer_reads; ++r) {
            for (size_t f = 0; f < pileup->featlen; ++f) {
                size_t old_coord = p * buffer_reads * pileup->featlen + r * pileup->featlen + f;
                if (old_coord < old_size) {
                    pileup->matrix[old_coord] = 0;
                }
            }
        }
    }
    pileup->buffer_reads = buffer_reads;
}

/** Destroys a pileup data structure.
 *
 *  @param data the object to cleanup.
 *  @returns void.
 *
 */
void destroy_read_aln_data(read_aln_data data) {
    free(data->matrix);
    free(data->major);
    free(data->minor);
    for (size_t r = 0; r < data->n_reads; ++r) {
        free(data->read_ids_left[r]);
        free(data->read_ids_right[r]);
    }
    free(data->read_ids_left);
    free(data->read_ids_right);
    free(data);
}

/** Prints a pileup data structure.
 *
 *  @param pileup a pileup structure.
 *  @returns void
 *
 */
void print_read_aln_data(read_aln_data pileup) {
    for (size_t p = 0; p < pileup->n_pos; ++p) {
        fprintf(stdout, "(pos, ins)\t");
        fprintf(stdout, "(%zu, %zu)\n", pileup->major[p], pileup->minor[p]);
        for (size_t r = 0; r < pileup->n_reads; ++r) {
            if (p == 0) {
                fprintf(stdout, "%s\t", pileup->read_ids_left[r]);
            }
            for (size_t f = 0; f < pileup->featlen; ++f) {
                int8_t c = pileup->matrix[p * pileup->buffer_reads * pileup->featlen +
                                          r * pileup->featlen + f];
                fprintf(stdout, "%i\t", c);
            }
            if (p == pileup->n_pos - 1) {
                fprintf(stdout, "%s\t", pileup->read_ids_right[r]);
            }
            fprintf(stdout, "\n");
        }
    }
}

/** Populate an array of dwells per base.
 *
 *  @param alignment an htslib alignment.
 *  @returns pointer to dewell array.
 */
int8_t *calculate_dwells(const bam1_t *alignment) {
    uint32_t length = alignment->core.l_qseq;

    int8_t *dwell_arr = xalloc(length, sizeof(int8_t), "dwell array");
    uint8_t *mv_tag = bam_aux_get(alignment, "mv");
    if (!mv_tag) {
        return dwell_arr;
    }
    uint32_t mv_len = bam_auxB_len(mv_tag);
    // uint8_t stride = bam_auxB2i(mv_tag, 0);

    size_t qpos = 0;  // base index

    if (alignment->core.flag & BAM_FREVERSE) {
        uint32_t dwell = 0;
        // Reversed alignment, iterate backward through move table.
        // Last entry is the first move which corresponds to
        // the last base
        for (size_t i = mv_len - 1; i > 0; --i) {
            ++dwell;
            if (bam_auxB2i(mv_tag, i) == 1) {
                dwell_arr[qpos] = (int8_t)min(dwell, __INT8_MAX__);
                ++qpos;
                dwell = 0;
            }
        }
    } else {
        uint32_t dwell = 1;
        // Skip first entry since this is always a move.
        // Last entry is the last sample point so need to
        // add the dwell since the last move afterwards
        for (size_t i = 2; i < mv_len; ++i) {
            if (bam_auxB2i(mv_tag, i) == 1) {
                dwell_arr[qpos] = (int8_t)min(dwell, __INT8_MAX__);
                ++qpos;
                dwell = 0;
            }
            ++dwell;
        }
        dwell_arr[qpos] = (int8_t)min(dwell, __INT8_MAX__);
    }
    return dwell_arr;
}

size_t aligned_ref_pos_from_cigar(uint32_t *cigar, uint32_t n_cigar) {
    uint32_t aligned_ref_pos = 0;
    for (size_t ci = 0; ci < n_cigar; ++ci) {
        uint32_t cigar_len = cigar[ci] >> 4;
        uint8_t cigar_op = cigar[ci] & 0xf;
        if ((cigar_op == BAM_CMATCH) || (cigar_op == BAM_CDEL) || (cigar_op == BAM_CEQUAL) ||
            (cigar_op == BAM_CDIFF)) {
            aligned_ref_pos += cigar_len;
        }
    }
    return aligned_ref_pos;
}

/** Generates medaka-style feature data in a region of a bam.
 *
 *  @param region 1-based region string.
 *  @param bam_file input aligment file.
 *  @param num_dtypes number of datatypes in bam.
 *  @param dtypes prefixes on query names indicating datatype.
 *  @param tag_name by which to filter alignments.
 *  @param tag_value by which to filter data.
 *  @param keep_missing alignments which do not have tag.
 *  @param read_group used for filtering.
 *  @param min_mapQ mininimum mapping quality.
 *  @param include_dwells whether to include dwells channel in features.
 *  @param include_haplotype whether to include haplotag channel in features.
 *  @param max_reads maximum allowed read depth.
 *  @returns a pileup data pointer.
 *
 *  The return value can be freed with destroy_read_aln_data.
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
read_aln_data calculate_read_alignment(const char *region,
                                       const bam_fset *bam_set,
                                       size_t num_dtypes,
                                       char *dtypes[],
                                       const char tag_name[2],
                                       const int tag_value,
                                       const bool keep_missing,
                                       const char *read_group,
                                       const int min_mapQ,
                                       const bool row_per_read,
                                       const bool include_dwells,
                                       const bool include_haplotype,
                                       const unsigned int max_reads) {
    if (num_dtypes == 1 && dtypes != NULL) {
        fprintf(stderr, "Recieved invalid num_dtypes and dtypes args.\n");
        exit(1);
    }
    // extract `chr`:`start`-`end` from `region`
    //   (start is one-based and end-inclusive),
    //   hts_parse_reg below sets return value to point
    //   at ":", copy the input then set ":" to null terminator
    //   to get `chr`.
    int start, end;
    char *chr = xalloc(strlen(region) + 1, sizeof(char), "chr");
    strcpy(chr, region);
    char *reg_chr = (char *)hts_parse_reg(chr, &start, &end);
    // start and end now zero-based end exclusive
    if (reg_chr) {
        *reg_chr = '\0';
    } else {
        fprintf(stderr, "Failed to parse region: '%s'.\n", region);
    }

    // open bam etc.
    // this is all now deferred to the caller
    htsFile *fp = bam_set->fp;
    hts_idx_t *idx = bam_set->idx;
    sam_hdr_t *hdr = bam_set->hdr;

    // setup bam interator
    mplp_data *data = xalloc(1, sizeof(mplp_data), "pileup init data");
    data->fp = fp;
    data->hdr = hdr;
    data->iter = bam_itr_querys(idx, hdr, region);
    data->min_mapQ = min_mapQ;
    memcpy(data->tag_name, tag_name, 2);
    data->tag_value = tag_value;
    data->keep_missing = keep_missing;
    data->read_group = read_group;

    bam_mplp_t mplp = bam_mplp_init(1, read_bam, (void **)&data);
    const bam_pileup1_t **plp = xalloc(1, sizeof(bam_pileup1_t *), "pileup");
    int ret, pos, tid, n_plp;

    // allocate output assuming one insertion per ref position
    size_t n_pos = 0;
    size_t max_n_reads = 0;
    size_t buffer_pos = 2 * (end - start);
    size_t buffer_reads = min(max_reads, 100);
    size_t extra_featlen =
            (include_dwells ? 1 : 0) + (include_haplotype ? 1 : 0) + (num_dtypes > 1 ? 1 : 0);
    read_aln_data pileup =
            create_read_aln_data(n_pos, max_n_reads, buffer_pos, buffer_reads, extra_featlen, 0);

    size_t major_col = 0;  // index into `pileup` corresponding to pos
    n_pos = 0;             // number of processed columns (including insertions)
    size_t min_gap = 5;    // minimum gap before starting a new read on an existing row

    // a kvec vector to store all read struct
    kvec_t(Read) read_array;
    kv_init(read_array);
    // hash map from read ids to index in above vector
    khash_t(str2int) *read_map = khash_str2int_init();

    while ((ret = bam_mplp_auto(mplp, &tid, &pos, &n_plp, plp) > 0)) {
        const char *c_name = data->hdr->target_name[tid];
        if (strcmp(c_name, chr) != 0) {
            continue;
        }
        if (pos < start) {
            continue;
        }
        if (pos >= end) {
            break;
        }
        n_pos++;

        // find maximum insert and number of reads
        int max_ins = 0;
        for (int i = 0; i < n_plp; ++i) {
            const bam_pileup1_t *p = plp[0] + i;
            if (p->indel > 0 && max_ins < p->indel) {
                max_ins = p->indel;
            }
        }
        if ((size_t)n_plp > max_n_reads) {
            max_n_reads = n_plp;
        }

        // reallocate output if necessary
        if (n_pos + max_ins > pileup->buffer_pos) {
            float cols_per_pos = (float)(n_pos + max_ins) / (1 + pos - start);
            // max_ins can dominate so add at least that
            buffer_pos = max_ins + max(2 * pileup->buffer_pos, (int)cols_per_pos * (end - start));
            enlarge_read_aln_data_pos(pileup, buffer_pos);
        }
        if ((pileup->buffer_reads < max_reads) &&
            (max_n_reads + (row_per_read ? n_plp : 0) > pileup->buffer_reads)) {
            buffer_reads = min(max_reads, max(max_n_reads + (row_per_read ? n_plp : 0),
                                              2 * pileup->buffer_reads));
            // correct start position of the column we're about to write
            major_col = (major_col / pileup->buffer_reads) * buffer_reads;
            enlarge_read_aln_data_reads(pileup, buffer_reads);
        }
        // set major/minor position indexes, minors hold ins
        for (int i = 0; i <= max_ins; ++i) {
            pileup->major[major_col / (pileup->featlen * pileup->buffer_reads) + i] = pos;
            pileup->minor[major_col / (pileup->featlen * pileup->buffer_reads) + i] = i;
        }

        // loop through all reads at this position
        for (int i = 0; i < n_plp; ++i) {
            const bam_pileup1_t *p = plp[0] + i;
            if (p->is_refskip) {
                continue;
            }

            const bam1_t *alignment = p->b;
            const char *qname = bam_get_qname(alignment);

            // check whether read is in hash list
            int read_i;
            int kh_rv = khash_str2int_get(read_map, qname, &read_i);
            if (kh_rv == -1) {  // a new read
                // get dtype tag
                size_t dtype = 0;
                bool failed = false;
                char *tag_val;
                uint8_t *tag;
                if (num_dtypes > 1) {
                    tag = bam_aux_get(alignment, datatype_tag);
                    if (tag == NULL) {  // tag isn't present
                        failed = true;
                    } else {
                        tag_val = bam_aux2Z(tag);
                        failed = errno == EINVAL;
                    }
                    if (!failed) {
                        bool found = false;
                        for (dtype = 0; dtype < num_dtypes; ++dtype) {
                            if (strcmp(dtypes[dtype], tag_val) == 0) {
                                found = true;
                                break;
                            }
                        }
                        failed = !found;
                    }
                    if (failed) {
                        fprintf(stderr, "Datatype not found for %s.\n", qname);
                        exit(1);
                    }
                }
                // get haplotype tag
                uint8_t haplotype = 0;
                failed = false;
                tag = bam_aux_get(alignment, "HP");
                if (tag == NULL) {  // tag isn't present
                    failed = true;
                } else {
                    haplotype = bam_aux2i(tag);
                    failed = errno == EINVAL;
                }

                Read read = {
                        .ref_start = alignment->core.pos,
                        .qname = bam_get_qname(alignment),
                        .seqi = bam_get_seq(alignment),
                        .qual = bam_get_qual(alignment),
                        .mq = alignment->core.qual,
                        .haplotype = haplotype,
                        .strand = bam_is_rev(alignment) ? -1 : 1,
                        .dtype = dtype,
                        .ref_end = alignment->core.pos +
                                   aligned_ref_pos_from_cigar(bam_get_cigar(alignment),
                                                              alignment->core.n_cigar),
                        .dwells = (include_dwells ? calculate_dwells(alignment) : NULL),
                };

                // insert read into read_array in place of a read that's already completed
                size_t array_size = kv_size(read_array);
                int kh_rv;
                if (!row_per_read) {
                    for (read_i = 0; read_i < array_size; ++read_i) {
                        Read *current_read = &kv_A(read_array, read_i);
                        if ((size_t)pos >= current_read->ref_end + min_gap) {
                            kv_A(read_array, read_i) = read;
                            kh_rv = khash_str2int_set(read_map, strdup(qname), read_i);
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
                    if (read_i < pileup->buffer_reads) {
                        kv_push(Read, read_array, read);
                    }
                    kh_rv = khash_str2int_set(read_map, strdup(qname), read_i);
                }
                if (kh_rv == -1) {
                    fprintf(stdout, "Error inserting read %s into hash map\n", qname);
                    exit(1);
                }
            }

            if (read_i >= pileup->buffer_reads) {
                continue;
            }

            Read *read = &kv_A(read_array, read_i);
            if (n_pos == 1) {
                pileup->read_ids_left[read_i] = strdup(read->qname);
            }

            int base_i, min_minor = 0;
            int max_minor = p->indel > 0 ? p->indel : 0;
            if (p->is_del) {
                pileup->matrix[major_col + pileup->featlen * read_i + 0] = del_val;       //base
                pileup->matrix[major_col + pileup->featlen * read_i + 1] = -1;            //qual
                pileup->matrix[major_col + pileup->featlen * read_i + 2] = read->strand;  //strand
                pileup->matrix[major_col + pileup->featlen * read_i + 3] = read->mq;      //mapq
                if (include_dwells) {
                    pileup->matrix[major_col + pileup->featlen * read_i + base_featlen] =
                            -1;  //dwell
                }
                if (include_haplotype) {
                    pileup->matrix[major_col + pileup->featlen * read_i + base_featlen +
                                   (include_dwells ? 1 : 0)] = read->haplotype;  //haplotag
                }
                if (num_dtypes > 1) {
                    pileup->matrix[major_col + pileup->featlen * read_i + base_featlen +
                                   (include_dwells ? 1 : 0) + (include_haplotype ? 1 : 0)] =
                            read->dtype;  //dtype
                }
                min_minor = 1;  // in case there is also an indel, skip the major position
            }
            // loop over any query bases at or inserted after pos
            int query_pos_offset = 0;
            int minor = min_minor;
            for (; minor <= max_minor; ++minor, ++query_pos_offset) {
                int base_j = bam1_seqi(read->seqi, p->qpos + query_pos_offset);
                base_i = num2countbase_symm[base_j];
                size_t partial_index =
                        major_col +
                        pileup->featlen * pileup->buffer_reads * minor  // skip to column
                        + pileup->featlen * read_i;                     // skip to read row

                pileup->matrix[partial_index + 0] = base_i;                                  //base
                pileup->matrix[partial_index + 1] = read->qual[p->qpos + query_pos_offset];  //qual
                pileup->matrix[partial_index + 2] = read->strand;  //strand
                pileup->matrix[partial_index + 3] = read->mq;      //qual
                if (include_dwells && read->dwells) {
                    pileup->matrix[partial_index + base_featlen] =
                            read->dwells[p->qpos + query_pos_offset];  //dwell
                }
                if (include_haplotype) {
                    pileup->matrix[partial_index + base_featlen + (include_dwells ? 1 : 0)] =
                            read->haplotype;  //haplotag
                }
                if (num_dtypes > 1) {
                    pileup->matrix[partial_index + base_featlen + (include_dwells ? 1 : 0) +
                                   (include_haplotype ? 1 : 0)] = read->dtype;  //dtype
                }
            }
            for (; minor <= max_ins; ++minor) {
                size_t partial_index =
                        major_col +
                        pileup->featlen * pileup->buffer_reads * minor  // skip to column
                        + pileup->featlen * read_i;                     // skip to read row

                pileup->matrix[partial_index + 0] = del_val;       //base
                pileup->matrix[partial_index + 1] = -1;            //qual
                pileup->matrix[partial_index + 2] = read->strand;  //strand
                pileup->matrix[partial_index + 3] = read->mq;      //qual
                if (include_dwells) {
                    pileup->matrix[partial_index + base_featlen] = -1;  //dwell
                }
                if (include_haplotype) {
                    pileup->matrix[partial_index + base_featlen + (include_dwells ? 1 : 0)] =
                            read->haplotype;  //haplotag
                }
                if (num_dtypes > 1) {
                    pileup->matrix[partial_index + base_featlen + (include_dwells ? 1 : 0) +
                                   (include_haplotype ? 1 : 0)] = read->dtype;  //dtype
                }
            }
        }
        major_col += (pileup->featlen * pileup->buffer_reads * (max_ins + 1));
        n_pos += max_ins;
    }

    for (size_t r = 0, nleft = 0, nright = 0; r < kv_size(read_array); ++r) {
        Read *read = &kv_A(read_array, r);
        if (read->ref_end >= (size_t)pos) {
            pileup->read_ids_right[r] = strdup(read->qname);
        } else {
            char tmp[100];
            sprintf(tmp, "__blank_%zu", ++nright);
            pileup->read_ids_right[r] = strdup(tmp);
        }
        if (!pileup->read_ids_left[r]) {
            char tmp[100];
            sprintf(tmp, "__blank_%zu", ++nleft);
            pileup->read_ids_left[r] = strdup(tmp);
        }
    }

    pileup->n_pos = n_pos;
    if (row_per_read) {
        pileup->n_reads = kv_size(read_array);
    } else {
        pileup->n_reads = max_n_reads;
    }
    pileup->n_reads = min(max_reads, pileup->n_reads);

    khash_str2int_destroy_free(read_map);
    kv_destroy(read_array);

    bam_itr_destroy(data->iter);
    bam_mplp_destroy(mplp);
    free(data);
    free(plp);
    free(chr);

    return pileup;
}
