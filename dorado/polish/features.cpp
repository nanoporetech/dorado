#include "features.h"

#include "polish/medaka_counts.h"

#include <spdlog/spdlog.h>

namespace dorado::polisher {

namespace {

CountsResult plp_data_to_tensors(PileupData& data, const size_t n_rows) {
    CountsResult result;

    // Create a tensor for the feature matrix (equivalent to np_counts in Python).
    // Torch tensors are row-major, so we create a tensor of size (n_cols, n_rows).
    result.counts = torch::from_blob(data.matrix().data(),
                                     {static_cast<long>(data.n_cols()), static_cast<long>(n_rows)},
                                     torch::kInt64)
                            .clone();

    // Create a tensor for the positions (equivalent to positions['major'] and positions['minor']).
    // We'll store the major and minor arrays as two separate columns in a single tensor.
    result.positions = torch::empty({static_cast<long>(data.n_cols()), 2}, torch::kInt64);

    // Copy 'major' data into the first column of the positions tensor.
    torch::Tensor major_tensor =
            torch::from_blob(data.major().data(), {static_cast<long>(data.n_cols())}, torch::kInt64)
                    .clone();
    result.positions.select(1, MAJOR_COLUMN).copy_(major_tensor);

    // Copy 'minor' data into the second column of the positions tensor.
    torch::Tensor minor_tensor =
            torch::from_blob(data.minor().data(), {static_cast<long>(data.n_cols())}, torch::kInt64)
                    .clone();
    result.positions.select(1, MINOR_COLUMN).copy_(minor_tensor);

    result.positions = result.positions.contiguous();

    return result;
}

/**
 * \brief Function to calculate feature vector normalization groups
 * \param dtypes Vector of data type names (strings).
 * \param num_qstrat Qscore stratifications.
 * \return Lookup of the form: key = (dtype, strand) -> vector of indices
 */
FeatureIndicesType pileup_counts_norm_indices(const std::vector<std::string>& dtypes,
                                              const size_t num_qstrat = 1) {
    // Create a map to store the indices.
    FeatureIndicesType indices;

    const int64_t plp_bases_size = static_cast<int64_t>(std::size(PILEUP_BASES));

    constexpr size_t featlen = std::size(PILEUP_BASES);

    // Iterate over each datatype.
    for (int64_t dti = 0; dti < static_cast<int64_t>(std::size(dtypes)); ++dti) {
        const std::string& dt = dtypes[dti];

        // Iterate over qscore stratification layers.
        for (int64_t qindex = 0; qindex < static_cast<int64_t>(num_qstrat); ++qindex) {
            // Iterate over the base codes (e.g., 'a', 'c', 'g', 't', etc.)
            for (int64_t base_i = 0; base_i < plp_bases_size; ++base_i) {
                const char code = PILEUP_BASES[base_i];
                const bool is_rev = std::islower(code);
                const int64_t index = base_i + dti * num_qstrat * featlen + qindex * featlen;
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
 * \param
 * \param
 * \param
 * \param
 * \param
 * \param
 * \param
 * \param
 * \param
 * \returns Vector of CountsResult objects. More than 1 object can be returned if the region
 *          was split internally. This can happen if there are discontinuities in positions
 *          caused e.g. by gaps in coverage.
 *
 * NOTE: The original implementation had another parameter here: `region_split=100000` which
 *          chunked the regions for parallel processing and returned these chunks separately in the end as well.
 *          Here, we move this responsibility onto the caller of the function.
 */
std::vector<CountsResult> construct_pileup_counts(bam_fset& bam_set,
                                                  const std::string& ref_name,
                                                  const int32_t ref_start,
                                                  const int32_t ref_end,
                                                  size_t num_qstrat = 1,
                                                  size_t num_dtypes = 1,
                                                  const std::vector<std::string>& dtypes = {},
                                                  const std::string tag_name = {},
                                                  int tag_value = 0,
                                                  bool keep_missing = false,
                                                  bool weibull_summation = false,
                                                  const char* read_group = NULL,
                                                  const int min_mapq = 1) {
    // Compute the pileup.
    // NOTE: the `num_qstrat` is passed into the `num_homop` parameter as is done in `pileup_counts` in features.py.
    // NOTE 2: the from_blob expects non-const data, so can't define the pileup as const here.
    PileupData pileup = calculate_pileup(ref_name, ref_start, ref_end, bam_set, num_dtypes, dtypes,
                                         num_qstrat, tag_name, tag_value, keep_missing,
                                         weibull_summation, read_group, min_mapq);
    // Create Torch tensors from the pileup.
    const size_t n_rows = std::size(PILEUP_BASES) * num_dtypes * num_qstrat;
    CountsResult counts_result = plp_data_to_tensors(pileup, n_rows);

    // print_pileup_data(pileup, num_dtypes, dtypes, num_qstrat);

    // destroy_plp_data(pileup);

    // // TODO: __enforce_pileup_chunk_contiguity
    // return enforce_pileup_chunk_contiguity(counts_result);

    const auto find_gaps = [](const torch::Tensor& positions,
                              int64_t threshold = 1) -> std::vector<int64_t> {
        const torch::Tensor diffs = (positions.size(0) >= 2)
                                            ? (positions.slice(0, 1) - positions.slice(0, 0, -1))
                                            : torch::empty({0}, positions.options());
        const auto gaps = (torch::nonzero(diffs > threshold).flatten() + 1).to(torch::kInt64);
        return std::vector<int64_t>(gaps.data_ptr<int64_t>(),
                                    gaps.data_ptr<int64_t>() + gaps.numel());
    };

    const auto split_on_discontinuities = [&find_gaps](const std::vector<CountsResult>& pileups) {
        std::vector<CountsResult> split_results;

        // TODO: Reimplement this with iteration over data instead of so much tensor slicing.
        for (const auto& data : pileups) {
            const auto positions_major =
                    data.positions.select(1, MAJOR_COLUMN);  // Accessing the 'major' column
            const std::vector<int64_t> gaps = find_gaps(positions_major);

            if (std::empty(gaps)) {
                split_results.emplace_back(data);
            } else {
                int64_t start = 0;
                for (const int64_t i : gaps) {
                    split_results.emplace_back(CountsResult{data.counts.slice(0, start, i),
                                                            data.positions.slice(0, start, i)});
                    start = i;
                }
                split_results.emplace_back(
                        CountsResult{data.counts.slice(0, start), data.positions.slice(0, start)});
            }
        }

        return split_results;
    };

    const auto merge_chunks = [](std::vector<CountsResult>& pileups) {
        std::vector<torch::Tensor> counts_buffer;
        std::vector<torch::Tensor> positions_buffer;
        int64_t last_major = -1;

        std::vector<CountsResult> results;

        for (auto& data : pileups) {
            if (!data.positions.size(0)) {
                continue;
            }
            const auto first_major = data.positions.select(1, MAJOR_COLUMN)[0].item<int64_t>();
            if (counts_buffer.empty() || (first_major - last_major) == 1) {
                // New or contiguous chunk.
                last_major = data.positions.select(1, MAJOR_COLUMN).index({-1}).item<int64_t>();
                counts_buffer.emplace_back(std::move(data.counts));
                positions_buffer.emplace_back(std::move(data.positions));

            } else {
                // Discontinuity found, finalize the current chunk
                last_major = data.positions.select(1, MAJOR_COLUMN).index({-1}).item<int64_t>();
                results.emplace_back(
                        CountsResult{concatenate(counts_buffer), concatenate(positions_buffer)});
                counts_buffer = {std::move(data.counts)};
                positions_buffer = {std::move(data.positions)};
            }
        }

        if (!counts_buffer.empty()) {
            results.emplace_back(
                    CountsResult{concatenate(counts_buffer), concatenate(positions_buffer)});
        }

        return results;
    };

    // First pass: split at discontinuities within each chunk.
    std::vector<CountsResult> split_results = split_on_discontinuities({counts_result});

    // Second pass: merge neighboring chunks if they have no distance between them.
    std::vector<CountsResult> results = merge_chunks(split_results);

    return results;
}

/**
 * \brief This is analogous to the `_post_process_pileup` function in Medaka.
 *
 * NOTE: This can update the pileup counts if `sym_indels == true`.
 */
Sample counts_to_features(CountsResult& pileup,
                          const std::string& ref_name,
                          const int64_t ref_start,
                          const int64_t ref_end,
                          const int32_t seq_id,
                          const int32_t win_id,
                          const bool sym_indels,
                          const FeatureIndicesType& feature_indices,
                          const NormaliseType normalise_type) {
    const int64_t start = pileup.positions.select(1, MAJOR_COLUMN).index({0}).item<int64_t>();
    const int64_t end = pileup.positions.select(1, MAJOR_COLUMN).index({-1}).item<int64_t>();

    if ((start != ref_start) || ((end + 1) != ref_end)) {
        spdlog::warn(
                "Pileup counts do not span requested region, requested {}:{}-{}, received {}-{}.",
                ref_name, ref_start, ref_end, start, end);
    }

    const auto minor_inds = torch::nonzero(pileup.positions.select(1, MINOR_COLUMN) > 0).squeeze();
    const auto major_pos_at_minor_inds = pileup.positions.index({minor_inds, MAJOR_COLUMN});
    const auto major_ind_at_minor_inds = torch::searchsorted(
            pileup.positions.select(1, MAJOR_COLUMN).contiguous(), major_pos_at_minor_inds, "left");

    auto depth = pileup.counts.sum(1);
    depth.index_put_({minor_inds}, depth.index({major_ind_at_minor_inds}));
    const auto depth_unsequezed = depth.unsqueeze(1).to(FeatureTensorType);

    if (sym_indels) {
        for (const auto& [key, inds] : feature_indices) {
            // const std::string& data_type = kv.first.first;
            const bool is_rev = key.second;
            const torch::Tensor inds_tensor = torch::tensor(inds, torch::dtype(torch::kInt64));

            const auto dt_depth =
                    pileup.counts.index({torch::indexing::Slice(), inds_tensor}).sum(1);
            // dt_depth.index_put_({minor_inds}, dt_depth.index({major_ind_at_minor_inds}));

            // Define deletion index.
            const int64_t featlen_index = is_rev ? PILEUP_POS_DEL_REV : PILEUP_POS_DEL_FWD;
            const int64_t dtype_size = std::size(PILEUP_BASES);

            // Find the deletion index
            // std::vector<int64_t> deletion_indices;
            for (const int64_t x : inds) {
                if ((x % dtype_size) == featlen_index) {
                    // deletion_indices.emplace_back(x);
                    pileup.counts.index_put_({minor_inds, x},
                                             dt_depth.index({major_ind_at_minor_inds}) -
                                                     dt_depth.index({minor_inds}));
                }
            }
            // // Ensure we have at least one valid deletion index
            // if (!deletion_indices.empty()) {
            //     del_ind = deletion_indices[0];  // Take the first valid index
            //     // Update counts for minor indices based on the calculated depths
            //     counts.index_put_({minor_inds, del_ind}, dt_depth.index({major_ind_at_minor_inds}) - dt_depth.index({minor_inds}));
            // } else {
            //     // Handle the case where no deletion index is found (optional)
            //     // e.g., log a warning or set a default behavior
            // }
        }
    }

    torch::Tensor feature_array;
    if (normalise_type == NormaliseType::TOTAL) {
        feature_array =
                pileup.counts / torch::max(depth_unsequezed, torch::ones_like(depth_unsequezed));

    } else if (normalise_type == NormaliseType::FWD_REV) {
        feature_array = torch::empty_like(pileup.counts, FeatureTensorType);
        for (const auto& kv : feature_indices) {
            const std::vector<int64_t>& inds = kv.second;
            const torch::Tensor inds_tensor = torch::tensor(inds, torch::dtype(torch::kInt64));

            auto dt_depth = pileup.counts.index({torch::indexing::Slice(), inds_tensor}).sum(1);
            dt_depth.index_put_({minor_inds}, dt_depth.index({major_ind_at_minor_inds}));
            feature_array.index_put_(
                    {torch::indexing::Slice(), inds_tensor},
                    pileup.counts.index({torch::indexing::Slice(), inds_tensor}) /
                            torch::max(depth_unsequezed, torch::ones_like(depth_unsequezed)));
        }
    } else {
        feature_array = pileup.counts;
        feature_array = feature_array.to(FeatureTensorType);
    }

    // Step 5: Create and return Sample object
    Sample sample{ref_name,  feature_array, pileup.positions, depth,
                  ref_start, ref_end,       seq_id,           win_id};

    // // Log the result
    // std::cerr << "Processed " << sample.ref_name << " (median depth "
    //           << torch::median(depth).item<float>() << ")" << std::endl;

    return sample;
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
                                                        const int64_t ref_end,
                                                        const int32_t seq_id,
                                                        const int32_t win_id) const {
    constexpr size_t num_qstrat = 1;
    constexpr bool weibull_summation = false;

    const int32_t num_dtypes = static_cast<int32_t>(std::size(m_dtypes)) + 1;
    const char* read_group_ptr = std::empty(m_read_group) ? nullptr : m_read_group.c_str();
    const std::string region =
            ref_name + ':' + std::to_string(ref_start + 1) + '-' + std::to_string(ref_end);

    std::vector<CountsResult> pileups = construct_pileup_counts(
            *m_bam_set, ref_name, ref_start + 1, ref_end, num_qstrat, num_dtypes, m_dtypes,
            m_tag_name, m_tag_value, m_tag_keep_missing, weibull_summation, read_group_ptr,
            m_min_mapq);

    std::vector<Sample> results;

    for (auto& data : pileups) {
        if (!data.counts.numel()) {
            spdlog::warn("Pileup-feature is zero-length for {} indicating no reads in this region.",
                         region);
            results.emplace_back(Sample{ref_name,
                                        {},
                                        std::move(data.positions),
                                        {},
                                        ref_start,
                                        ref_end,
                                        seq_id,
                                        win_id});
            continue;
        }

        // results.emplace_back(Sample{ref_name, data.counts,
        //                             data.positions, {}});

        results.emplace_back(counts_to_features(data, ref_name, ref_start + 1, ref_end, seq_id,
                                                win_id, m_symmetric_indels, m_feature_indices,
                                                m_normalise_type));
    }

    return results;
}

std::vector<ConsensusResult> CountsFeatureEncoder::decode_bases(const torch::Tensor& logits,
                                                                const bool with_probs) const {
    static constexpr std::string_view label_scheme{"*ACGT"};

    const auto indices = logits.argmax(-1);  // Shape becomes [N, L]

    std::vector<ConsensusResult> results(indices.size(0));

    for (int64_t sample_id = 0; sample_id < indices.size(0); ++sample_id) {
        const auto& positions = indices[sample_id];

        std::string& seq = results[sample_id].seq;
        seq.resize(positions.size(0), '*');

        for (int64_t j = 0; j < positions.size(0); ++j) {
            const int64_t class_index = positions[j].item<int64_t>();
            assert(class_index < static_cast<int64_t>(std::size(label_scheme)));
            seq[j] = label_scheme[class_index];
        }
    }

    if (with_probs) {
        const torch::Tensor probs = torch::gather(logits, -1, indices.unsqueeze(-1)).squeeze(-1);

        // std::cerr << "probs: " << probs << "\n";

        for (int64_t sample_id = 0; sample_id < indices.size(0); ++sample_id) {
            std::string& quals = results[sample_id].quals;
            quals.clear();

            const auto phred_scores =
                    (-10.0 * torch::log10(1.0 - probs[sample_id])).clamp(0, 40).to(torch::kUInt8) +
                    33;

            quals.resize(phred_scores.size(0), '!');
            for (int64_t j = 0; j < phred_scores.size(0); ++j) {
                quals[j] = static_cast<char>(phred_scores[j].item<uint8_t>());
            }
        }
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
