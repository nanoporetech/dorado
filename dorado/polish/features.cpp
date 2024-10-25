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
    result.positions.select(1, MAJOR_COLUMN).copy_(major_tensor);

    // Copy 'minor' data into the second column of the positions tensor.
    torch::Tensor minor_tensor =
            torch::from_blob(data->minor, {static_cast<long>(data->n_cols)}, torch::kInt64).clone();
    result.positions.select(1, MINOR_COLUMN).copy_(minor_tensor);

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

/**
 * \brief Creates pileup counts for feature array for a given region.
 * \param bam_set File stream for the open BAM file for random access lookup.
 * \param region Htslib-style region string. Start is 1-based, and end is inclusive.
 * \returns Vector of CountsResult objects. More than 1 object can be returned if the region
 *          was split internally. This can happen if there are discontinuities in positions
 *          caused e.g. by gaps in coverage.
 *
 * NOTE: The original implementation had another parameter here: `region_split=100000` which
 *          chunked the regions for parallel processing and returned these chunks separately in the end as well.
 *          Here, we move this responsibility onto the caller of the function.
 */
std::vector<CountsResult> construct_pileup_counts(bam_fset* bam_set,
                                                  const std::string& region,
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
    std::cout << "[construct_pileup_counts] region = " << region << "\n";
    const plp_data pileup = calculate_pileup(region.c_str(), bam_set, num_dtypes, dtypes,
                                             num_qstrat, tag_name.c_str(), tag_value, keep_missing,
                                             weibull_summation, read_group, min_mapQ);
    // Create Torch tensors from the pileup.
    const size_t n_rows = featlen * num_dtypes * num_qstrat;
    CountsResult counts_result = plp_data_to_tensors(pileup, n_rows);

    print_pileup_data(pileup, num_dtypes, dtypes, num_qstrat);

    destroy_plp_data(pileup);

    // // TODO: __enforce_pileup_chunk_contiguity
    // return enforce_pileup_chunk_contiguity(counts_result);

    // const auto split_on_discontinuities = [](const std::vector<CountsResult>& pileups) {
    //     std::vector<CountsResult> split_results;

    //     // First pass: split at discontinuities within each chunk
    //     for (const auto& data: pileups) {
    //         auto positions_major = data.positions.select(1, 0);  // Accessing the 'major' column
    //         // auto move = positions_major.slice(0, 1) - positions_major.slice(0, 0, -1);
    //         [[maybe_unused]] torch::Tensor move = positions_major.slice(0, 1) - positions_major.slice(0, 0, -1);
    //         // auto gaps = find_gaps(move);

    //         split_results.emplace_back(data);

    //         // if (gaps.empty()) {
    //         //     split_results.emplace_back(counts, positions);
    //         // } else {
    //         //     int64_t start = 0;
    //         //     for (auto i : gaps) {
    //         //         split_results.emplace_back(
    //         //             counts.slice(0, start, i),
    //         //             positions.slice(0, start, i));
    //         //         start = i;
    //         //     }
    //         //     split_results.emplace_back(
    //         //         counts.slice(0, start),
    //         //         positions.slice(0, start));
    //         // }
    //     }

    //     return split_results;
    // };

    // return split_on_discontinuities({counts_result});

    return {counts_result};
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

std::vector<Sample> CountsFeatureEncoder::encode_region(const std::string& ref_name,
                                                        const int64_t ref_start,
                                                        const int64_t ref_end) {
    constexpr size_t num_qstrat = 1;
    constexpr bool weibull_summation = false;

    std::vector<const char*> dtypes;
    for (const auto& dtype : m_dtypes) {
        dtypes.emplace_back(dtype.c_str());
    }

    const char** dtypes_ptr = std::empty(dtypes) ? nullptr : dtypes.data();
    const int32_t num_dtypes = static_cast<int32_t>(std::size(dtypes)) + 1;
    const char* read_group_ptr = std::empty(m_read_group) ? nullptr : m_read_group.c_str();
    const std::string region =
            ref_name + ':' + std::to_string(ref_start + 1) + '-' + std::to_string(ref_end);

    std::vector<CountsResult> pileups = construct_pileup_counts(
            m_bam_set, region, num_qstrat, num_dtypes, dtypes_ptr, m_tag_name.c_str(), m_tag_value,
            m_tag_keep_missing, weibull_summation, read_group_ptr, m_min_mapq);

    static constexpr int32_t depth = 0;

    std::vector<Sample> results;

    for (auto& data : pileups) {
        results.emplace_back(Sample{ref_name, ref_start, ref_end, std::move(data.feature_matrix),
                                    std::move(data.positions), depth});
    }

    return results;
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
