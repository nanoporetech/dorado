#include "features.h"

#include "polish/medaka_counts.h"

#include <spdlog/spdlog.h>

namespace dorado::polisher {

void counts_feature_encoder(const bam_fset* bam_set, const std::string_view region) {
    size_t num_dtypes = 1;
    char** dtypes = NULL;
    char tag_name[2] = "";
    int tag_value = 0;
    bool keep_missing = false;
    size_t num_homop = 1;
    bool weibull_summation = false;
    const char* read_group = NULL;
    const int min_mapQ = 1;

    plp_data pileup =
            calculate_pileup(region.data(), bam_set, num_dtypes, dtypes, num_homop, tag_name,
                             tag_value, keep_missing, weibull_summation, read_group, min_mapQ);

    print_pileup_data(pileup, num_dtypes, dtypes, num_homop);
    fprintf(stdout, "pileup is length %zu, with buffer of %zu columns\n", pileup->n_cols,
            pileup->buffer_cols);
    destroy_plp_data(pileup);
}

}  // namespace dorado::polisher
