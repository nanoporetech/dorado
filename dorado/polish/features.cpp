#include "features.h"

#include "polish/medaka_counts.h"

#include <spdlog/spdlog.h>

namespace dorado::polisher {

namespace {

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

/**
 * \brief Function to calculate feature vector normalization groups
 * \param dtypes Vector of data type names (strings).
 * \param num_qstrat Qscore stratifications.
 * \return Lookup of the form: key = (dtype, strand) -> vector of indices
 */
std::unordered_map<std::pair<std::string, bool>, std::vector<size_t>, KeyHash>
pileup_counts_norm_indices(const std::vector<std::string>& dtypes, const size_t num_qstrat = 1) {
    // Create a map to store the indices.
    std::unordered_map<std::pair<std::string, bool>, std::vector<size_t>, KeyHash> indices;

    const size_t plp_bases_size = featlen;  // TODO: plp_bases.size()

    // Iterate over each datatype.
    for (size_t dti = 0; dti < dtypes.size(); ++dti) {
        const std::string& dt = dtypes[dti];

        // Iterate over qscore stratification layers.
        for (size_t qindex = 0; qindex < num_qstrat; ++qindex) {
            // Iterate over the base codes (e.g., 'a', 'c', 'g', 't', etc.)
            for (size_t base_i = 0; base_i < plp_bases_size; ++base_i) {
                const char code = plp_bases[base_i];
                const bool is_rev = std::islower(code);
                const size_t index = base_i + dti * num_qstrat * featlen + qindex * featlen;
                indices[std::make_pair(dt, is_rev)].push_back(index);
            }
        }
    }

    return indices;
}

CountsResult construct_pileup_counts(bam_fset* bam_set,
                                     const std::string_view region,
                                     size_t num_qstrat = 1,
                                     size_t num_dtypes = 1,
                                     const char** dtypes = NULL,
                                     const std::string tag_name = {},
                                     //  const std::string_view tag_name = {},
                                     int tag_value = 0,
                                     bool keep_missing = false,
                                     //  size_t num_homop = 1,
                                     bool weibull_summation = false,
                                     const char* read_group = NULL,
                                     const int min_mapQ = 1) {
    // Compute the pileup.
    // NOTE: the `num_qstrat` is passed into the `num_homop` parameter as is done in `pileup_counts` in features.py.
    const plp_data pileup = calculate_pileup(region.data(), bam_set, num_dtypes, dtypes, num_qstrat,
                                             tag_name.data(), tag_value, keep_missing,
                                             weibull_summation, read_group, min_mapQ);
    // Create Torch tensors from the pileup.
    const size_t n_rows = featlen * num_dtypes * num_qstrat;
    CountsResult result = plp_data_to_tensors(pileup, n_rows);

    destroy_plp_data(pileup);

    return result;
}

}  // namespace

CountsFeatureEncoder::CountsFeatureEncoder(bam_fset* bam_set) : m_bam_set{bam_set} {}

CountsFeatureEncoder::CountsFeatureEncoder(bam_fset* bam_set,
                                           const NormaliseType normalise_type,
                                           const std::vector<std::string>& dtypes,
                                           const std::string_view tag_name,
                                           const int32_t tag_value,
                                           const bool tag_keep_missing,
                                           const std::string_view read_group,
                                           const int32_t min_mapq,
                                           const bool symmetric_indels)
        : m_bam_set{bam_set},
          m_normalise_type{normalise_type},
          m_dtypes{dtypes},
          m_tag_name{tag_name},
          m_tag_value{tag_value},
          m_tag_keep_missing{tag_keep_missing},
          m_read_group{read_group},
          m_min_mapq{min_mapq},
          m_symmetric_indels{symmetric_indels},
          m_feature_indices{pileup_counts_norm_indices(dtypes)} {}

CountsResult CountsFeatureEncoder::encode_region(const std::string_view region) {
    constexpr size_t num_qstrat = 1;
    constexpr bool weibull_summation = false;

    std::vector<const char*> dtypes;
    for (const auto& dtype : m_dtypes) {
        dtypes.emplace_back(dtype.c_str());
    }

    const char* read_group_ptr = std::empty(m_read_group) ? nullptr : m_read_group.c_str();
    const char** dtypes_ptr = std::empty(dtypes) ? nullptr : dtypes.data();
    // const char* tag_name_ptr = std::empty(m_tag_name) ? nullptr: m_tag_name.c_str();

    CountsResult result = construct_pileup_counts(
            m_bam_set, region, num_qstrat, dtypes.size(), dtypes_ptr, m_tag_name.c_str(),
            m_tag_value, m_tag_keep_missing, weibull_summation, read_group_ptr, m_min_mapq);

    return result;
}

// CountsResult counts_feature_encoder(bam_fset* bam_set, const std::string_view region) {
//     // TODO: Make sure which of these need to be parametrized to emulate `medaka inference`.
//     // Parameters for the pileup.
//     const size_t num_qstrat = 1;
//     const size_t num_dtypes = 1;
//     const bool weibull_summation = false;

//     // Parameters for the CountsFeatureEncoder.
//     // const NormaliseType normalise_type{NormaliseType::TOTAL};
//     const char** dtypes = NULL;
//     // std::string_view tag_name;
//     const std::string tag_name;
//     const int32_t tag_value = 0;
//     const bool tag_keep_missing = false;
//     const char* read_group = NULL;
//     const int min_mapQ = 1;
//     // const bool symmetric_indels = false;
//     // feature_indices = pileup_counts_norm_indices(self.dtypes)

//     CountsResult result = construct_pileup_counts(
//             bam_set, region, num_qstrat, num_dtypes, dtypes, tag_name,
//             tag_value, tag_keep_missing, weibull_summation, read_group, min_mapQ);

//     return result;
// }

}  // namespace dorado::polisher
