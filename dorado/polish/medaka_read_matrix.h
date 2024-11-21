#ifndef _MEDAKA_READ_ALN_H
#define _MEDAKA_READ_ALN_H

// medaka-style feature data
typedef struct _read_aln_data {
    size_t buffer_pos;
    size_t buffer_reads;
    size_t num_dtypes;
    size_t n_pos;
    size_t n_reads;
    size_t featlen;
    int8_t *matrix;
    size_t *major;
    size_t *minor;
    char **read_ids_left;
    char **read_ids_right;
} _read_aln_data;
typedef _read_aln_data *read_aln_data;

typedef struct Read {
    size_t ref_start;
    char *qname;
    uint8_t *seqi;
    uint8_t *qual;
    int8_t mq;
    size_t haplotype;
    int8_t strand;
    int8_t dtype;
    uint32_t ref_end;
    int8_t *dwells;
} Read;

// medaka-style base encoding
static const size_t base_featlen = 4;  // minimal number of feature channels
static const size_t del_val = 5;       // value representing deletion in base channel

// convert 16bit IUPAC (+16 for strand) to plp_bases index
static const int num2countbase_symm[32] = {
        -1, 1, 2, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1,
        -1, 1, 2, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1,
};

/** Constructs a pileup data structure.
 *
 *  @param n_pos number of pileup columns.
 *  @param n_reads number of pileup rows.
 *  @param buffer_pos number of pileup columns.
 *  @param buffer_reads number of pileup rows.
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
                                   size_t fixed_size);

/** Enlarge the internal buffers of a pileup data structure.
 *
 *  @param pileup a read_aln_data pointer.
 *  @param buffer_pos number of pileup columns for which to allocate memory
 *
 */
void enlarge_read_aln_data_pos(read_aln_data pileup, size_t buffer_pos);
/** Enlarge the internal buffers of a pileup data structure.
 *
 *  @param pileup a read_aln_data pointer.
 *  @param buffer_reads number of pileup rows for which to allocate memory
 *
 */
void enlarge_read_aln_data_reads(read_aln_data pileup, size_t buffer_reads);

/** Destroys a pileup data structure.
 *
 *  @param data the object to cleanup.
 *  @returns void.
 *
 */
void destroy_read_aln_data(read_aln_data data);

/** Prints a pileup data structure.
 *
 *  @param pileup a pileup counts structure.
 *  @returns void
 *
 */
void print_read_aln_data(read_aln_data pileup);

/** Generates medaka-style feature data in a region of a bam.
 *
 *  @param region 1-based region string.
 *  @param bam_file input aligment file.
 *  @param num_dtypes number of datatypes in bam.
 *  @param dtypes prefixes on query names indicating datatype.
 *  @param tag_name by which to filter alignments
 *  @param tag_value by which to filter data
 *  @param keep_missing alignments which do not have tag.
 *  @param read_group used for filtering.
 *  @param min_mapQ mininimum mapping quality.
 *  @param row_per_read place each new read on a new row.
 *  @param include_dwells whether to include dwells channel in features.
 *  @param include_haplotype whether to include haplotag channel in features.
 *  @param max_reads maximum allowed read depth.
 *  @returns a pileup counts data pointer.
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
 *  If row_per_read is False, new reads will be placed in the first row where 
 *  there previous read has terminated, if one exists.
 */
read_aln_data calculate_read_alignment(const char *region,
                                       const bam_fset *bam_set,
                                       size_t num_dtypes,
                                       char *dtypes[],
                                       const char tag_name[2],
                                       const int tag_value,
                                       const _Bool keep_missing,
                                       const char *read_group,
                                       const int min_mapQ,
                                       const _Bool row_per_read,
                                       const _Bool include_dwells,
                                       const _Bool include_haplotype,
                                       const unsigned int max_reads);

#endif
