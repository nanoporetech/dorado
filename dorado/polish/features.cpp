#include "features.h"

#include "polish/medaka_counts.h"

#include <spdlog/spdlog.h>

namespace dorado::polisher {

CountsResult plp_data_to_tensors(const plp_data& data, const size_t n_rows) {
    CountsResult result;

    // Create a tensor for the feature matrix (equivalent to np_counts in Python).
    // Torch tensors are row-major, so we create a tensor of size (n_cols, n_rows).
    result.feature_matrix =
            torch::from_blob(data->matrix,
                             {static_cast<long>(data->n_cols), static_cast<long>(n_rows)},
                             torch::kInt64)
                    .clone();

    // Create a tensor for the positions (equivalent to positions['major'] and positions['minor']).
    // We'll store the major and minor arrays as two separate columns in a single tensor.
    result.positions = torch::empty({static_cast<long>(data->n_cols), 2}, torch::kInt64);

    // Copy 'major' data into the first column of the positions tensor.
    torch::Tensor major_tensor =
            torch::from_blob(data->major, {static_cast<long>(data->n_cols)}, torch::kInt64).clone();
    result.positions.select(1, 0).copy_(major_tensor);

    // Copy 'minor' data into the second column of the positions tensor.
    torch::Tensor minor_tensor =
            torch::from_blob(data->minor, {static_cast<long>(data->n_cols)}, torch::kInt64).clone();
    result.positions.select(1, 1).copy_(minor_tensor);

    return result;
}

CountsResult construct_pileup_counts(const bam_fset* bam_set,
                                     const std::string_view region,
                                     size_t num_qstrat = 1,
                                     size_t num_dtypes = 1,
                                     char** dtypes = NULL,
                                     const std::string_view tag_name = {},
                                     int tag_value = 0,
                                     bool keep_missing = false,
                                     size_t num_homop = 1,
                                     bool weibull_summation = false,
                                     const char* read_group = NULL,
                                     const int min_mapQ = 1) {
    // Compute the pileup.
    const plp_data pileup =
            calculate_pileup(region.data(), bam_set, num_dtypes, dtypes, num_homop, tag_name.data(),
                             tag_value, keep_missing, weibull_summation, read_group, min_mapQ);
    // Create Torch tensors from the pileup.
    const size_t n_rows = featlen * num_dtypes * num_qstrat;
    CountsResult result = plp_data_to_tensors(pileup, n_rows);

    destroy_plp_data(pileup);

    return result;
}

CountsResult counts_feature_encoder(const bam_fset* bam_set, const std::string_view region) {
    // TODO: Make sure which of these need to be parametrized to emulate `medaka inference`.
    size_t num_qstrat = 1;
    size_t num_dtypes = 1;
    char** dtypes = NULL;
    char tag_name[2] = "";
    int tag_value = 0;
    bool keep_missing = false;
    size_t num_homop = 1;
    bool weibull_summation = false;
    const char* read_group = NULL;
    const int min_mapQ = 1;
    // feature_indices = pileup_counts_norm_indices(self.dtypes)

    CountsResult result = construct_pileup_counts(
            bam_set, region, num_qstrat, num_dtypes, dtypes, std::string_view(tag_name, 2),
            tag_value, keep_missing, num_homop, weibull_summation, read_group, min_mapQ);

    return result;
}

}  // namespace dorado::polisher
