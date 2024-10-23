#ifndef _MEDAKA_COUNTS_H
#define _MEDAKA_COUNTS_H

// medaka-style feature data
typedef struct _plp_data {
    size_t buffer_cols;
    size_t num_dtypes;
    size_t num_homop;
    size_t n_cols;
    size_t *matrix;
    size_t *major;
    size_t *minor;
} _plp_data;
typedef _plp_data *plp_data;

/** Format an array values as a comma seperate string
 *
 * @param values integer input array
 * @param length size of input array
 * @param result output char buffer of size 4 * length * sizeof char
 * @returns void
 *
 * The output buffer size comes from:
 *    a single value is max 3 chars
 *    + 1 for comma (or \0 at end)
 */
void format_uint8_array(uint8_t *values, size_t length, char *result);

// Simple container for strings
typedef struct string_set {
    size_t n;
    char **strings;
} string_set;

/** Destroys a string set
 *
 *  @param data the object to cleanup.
 *  @returns void.
 *
 */
void destroy_string_set(string_set strings);

/** Retrieves contents of key-value tab delimited file.
 *
 *  @param fname input file path.
 *  @returns a string_set
 *
 *  The return value can be free'd with destroy_string_set.
 *  key-value pairs are stored sequentially in the string set
 *
 */
string_set read_key_value(char *fname);

// medaka-style base encoding
static const char plp_bases[] = "acgtACGTdD";
static const size_t featlen = 10;  // len of the above
static const size_t fwd_del = 9;   // position of D
static const size_t rev_del = 8;   // position of d

// bam tag used for datatypes
static const char datatype_tag[] = "DT";

// convert 16bit IUPAC (+16 for strand) to plp_bases index
static const int num2countbase[32] = {
        -1, 4, 5, -1, 6, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1,
        -1, 0, 1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1,
};

/** Constructs a pileup data structure.
 *
 *  @param n_cols number of pileup columns.
 *  @param buffer_cols number of pileup columns.
 *  @param num_dtypes number of datatypes in pileup.
 *  @param num_homop maximum homopolymer length to consider.
 *  @param fixed_size if not zero data matrix is allocated as fixed_size * n_cols, ignoring other arguments
 *  @see destroy_plp_data
 *  @returns a plp_data pointer.
 *
 *  The return value can be freed with destroy_plp_data.
 *
 */
plp_data create_plp_data(size_t n_cols,
                         size_t buffer_cols,
                         size_t num_dtypes,
                         size_t num_homop,
                         size_t fixed_size);

/** Enlarge the internal buffers of a pileup data structure.
 *
 *  @param pileup a plp_data pointer.
 *  @param buffer_cols number of pileup columns for which to allocate memory
 *
 */
void enlarge_plp_data(plp_data pileup, size_t buffer_cols);

/** Destroys a pileup data structure.
 *
 *  @param data the object to cleanup.
 *  @returns void.
 *
 */
void destroy_plp_data(plp_data data);

/** Prints a pileup data structure.
 *
 *  @param pileup a pileup counts structure.
 *  @param num_dtypes number of datatypes in the pileup.
 *  @param dtypes datatype prefix strings.
 *  @param num_homop maximum homopolymer length to consider.
 *  @returns void
 *
 */
void print_pileup_data(plp_data pileup, size_t num_dtypes, char *dtypes[], size_t num_homop);

/** Generates medaka-style feature data in a region of a bam.
 *
 *  @param region 1-based region string.
 *  @param bam_file input aligment file.
 *  @param num_dtypes number of datatypes in bam.
 *  @param dtypes prefixes on query names indicating datatype.
 *  @param num_homop maximum homopolymer length to consider.
 *  @param tag_name by which to filter alignments
 *  @param tag_value by which to filter data
 *  @param keep_missing alignments which do not have tag
 *  @param weibull_summation use predefined bam tags to perform homopolymer partial counts.
 *  @param read group used for filtering. 
 *  @param mininimum mapping quality for filtering. 
 *  @returns a pileup counts data pointer.
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
plp_data calculate_pileup(const char *region,
                          const bam_fset *bam_set,
                          size_t num_dtypes,
                          char *dtypes[],
                          size_t num_homop,
                          const char tag_name[2],
                          const int tag_value,
                          const _Bool keep_missing,
                          bool weibull_summation,
                          const char *read_group,
                          const int min_mapQ);

#endif
