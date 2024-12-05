#include "polish/architectures/encoder_counts.h"

#include "polish/medaka_counts.h"

#include <spdlog/spdlog.h>
#include <utils/timer_high_res.h>

namespace dorado::polisher {

namespace {

CountsResult plp_data_to_tensors(PileupData& data, const size_t n_rows) {
    const size_t num_bytes = static_cast<size_t>(data.n_cols() * n_rows * sizeof(int64_t));

    if (num_bytes == 0) {
        return {};
    }

    CountsResult result;

    // Allocate a tensor of the appropriate size directly for `result.counts` on the CPU
    result.counts = torch::empty({static_cast<long>(data.n_cols()), static_cast<long>(n_rows)},
                                 torch::kInt64);

    assert(result.counts.data_ptr<int64_t>() != nullptr);

    // Copy the data from `data.matrix()` into `result.counts`
    std::memcpy(result.counts.data_ptr<int64_t>(), data.get_matrix().data(), num_bytes);

    result.positions_major = std::move(data.get_major());
    result.positions_minor = std::move(data.get_minor());

    return result;
}

/**
 * \brief Function to calculate feature vector normalization groups
 * \param dtypes Vector of data type names (strings).
 * \param num_qstrat Qscore stratifications.
 * \return Lookup of the form: key = (dtype, strand) -> vector of indices
 */
FeatureIndicesType pileup_counts_norm_indices(const std::vector<std::string>& dtypes,
                                              const size_t num_qstrat) {
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

Sample counts_to_features(CountsResult& pileup,
                          const int32_t seq_id,
                          const bool sym_indels,
                          const FeatureIndicesType& feature_indices,
                          const NormaliseType normalise_type) {
    // Avoid slow Torch operations as much as possible. The original Medaka code had this implemented
    // on a very high level with lots of redundancy in computation.
    const int64_t num_rows = static_cast<int64_t>(std::size(pileup.positions_major));
    std::vector<int64_t> minor_inds;
    std::vector<int64_t> major_pos_at_minor_inds;
    std::vector<int64_t> major_ind_at_minor_inds;
    minor_inds.reserve(num_rows);
    major_pos_at_minor_inds.reserve(num_rows);
    major_ind_at_minor_inds.reserve(num_rows);
    int64_t last_non_minor_index = -1;
    for (int64_t i = 0; i < num_rows; ++i) {
        if (pileup.positions_minor[i] > 0) {
            minor_inds.emplace_back(i);
            major_pos_at_minor_inds.emplace_back(pileup.positions_major[i]);
            major_ind_at_minor_inds.emplace_back(last_non_minor_index);
        } else {
            last_non_minor_index = i;
        }
    }

    auto depth = pileup.counts.sum(1);

    auto depth_data = depth.data_ptr<int64_t>();
    for (size_t i = 0; i < std::size(minor_inds); ++i) {
        if (major_ind_at_minor_inds[i] != -1) {
            depth_data[minor_inds[i]] = depth_data[major_ind_at_minor_inds[i]];
        }
    }
    const auto depth_unsequezed = depth.unsqueeze(1).to(FeatureTensorType);

    if (sym_indels) {
        const torch::Tensor minor_inds_tensor = torch::tensor(minor_inds, torch::kInt64);
        const torch::Tensor major_ind_at_minor_inds_tensor =
                torch::tensor(major_ind_at_minor_inds, torch::kInt64);

        for (const auto& [key, inds] : feature_indices) {
            // const std::string& data_type = kv.first.first;
            const bool is_rev = key.second;
            const torch::Tensor inds_tensor = torch::tensor(inds, torch::dtype(torch::kInt64));

            const auto dt_depth =
                    pileup.counts.index({torch::indexing::Slice(), inds_tensor}).sum(1);
            // dt_depth.index_put_({minor_inds}, dt_depth.index({major_ind_at_minor_inds}));

            // Define deletion index.
            const int64_t featlen_index = is_rev ? PILEUP_POS_DEL_REV : PILEUP_POS_DEL_FWD;
            const int64_t dtype_size = PILEUP_BASES_SIZE;

            // Find the deletion index
            // std::vector<int64_t> deletion_indices;
            for (const int64_t x : inds) {
                if ((x % dtype_size) == featlen_index) {
                    // deletion_indices.emplace_back(x);
                    pileup.counts.index_put_({minor_inds_tensor, x},
                                             dt_depth.index({major_ind_at_minor_inds_tensor}) -
                                                     dt_depth.index({minor_inds_tensor}));
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
        const torch::Tensor minor_inds_tensor = torch::tensor(minor_inds, torch::kInt64);
        const torch::Tensor major_ind_at_minor_inds_tensor =
                torch::tensor(major_ind_at_minor_inds, torch::kInt64);
        for (const auto& kv : feature_indices) {
            const std::vector<int64_t>& inds = kv.second;
            const torch::Tensor inds_tensor = torch::tensor(inds, torch::dtype(torch::kInt64));

            auto dt_depth = pileup.counts.index({torch::indexing::Slice(), inds_tensor}).sum(1);
            dt_depth.index_put_({minor_inds_tensor},
                                dt_depth.index({major_ind_at_minor_inds_tensor}));
            feature_array.index_put_(
                    {torch::indexing::Slice(), inds_tensor},
                    pileup.counts.index({torch::indexing::Slice(), inds_tensor}) /
                            torch::max(depth_unsequezed, torch::ones_like(depth_unsequezed)));
        }
    } else {
        feature_array = std::move(pileup.counts);
        feature_array = feature_array.to(FeatureTensorType);
    }

    Sample sample{std::move(feature_array),
                  std::move(pileup.positions_major),
                  std::move(pileup.positions_minor),
                  std::move(depth),
                  seq_id,
                  -1,
                  {},
                  {}};

    return sample;
}

std::vector<polisher::Sample> merge_adjacent_samples_impl(std::vector<polisher::Sample> samples) {
    std::vector<torch::Tensor> features_buffer;
    std::vector<std::vector<int64_t>> positions_major_buffer;
    std::vector<std::vector<int64_t>> positions_minor_buffer;
    std::vector<torch::Tensor> depth_buffer;
    int32_t seq_id_buffer = -1;
    int32_t region_id_buffer = -1;
    int64_t last_end = -1;

    std::vector<polisher::Sample> results;

    const auto cat_vectors = [](const std::vector<std::vector<int64_t>>& vecs) {
        size_t size = 0;
        for (const auto& vec : vecs) {
            size += std::size(vec);
        }
        std::vector<int64_t> ret;
        ret.reserve(size);
        for (const auto& vec : vecs) {
            ret.insert(std::end(ret), std::cbegin(vec), std::cend(vec));
        }
        return ret;
    };

    for (auto& sample : samples) {
        if (std::empty(sample.positions_major)) {
            continue;
        }
        const int64_t start = sample.start();

        if (std::empty(features_buffer) ||
            ((sample.seq_id == seq_id_buffer) && (sample.region_id == region_id_buffer) &&
             ((start - last_end) == 0))) {
            // New or contiguous chunk.
            last_end = sample.end();
            features_buffer.emplace_back(std::move(sample.features));
            positions_major_buffer.emplace_back(std::move(sample.positions_major));
            positions_minor_buffer.emplace_back(std::move(sample.positions_minor));
            depth_buffer.emplace_back(std::move(sample.depth));
            seq_id_buffer = sample.seq_id;
            region_id_buffer = sample.region_id;

        } else {
            // Discontinuity found, finalize the current chunk
            last_end = sample.end();

            // The torch::cat is slow, so just move if there is nothing to concatenate.
            if (std::size(features_buffer) == 1) {
                results.emplace_back(polisher::Sample{std::move(features_buffer.front()),
                                                      std::move(positions_major_buffer.front()),
                                                      std::move(positions_minor_buffer.front()),
                                                      std::move(depth_buffer.front()),
                                                      seq_id_buffer,
                                                      region_id_buffer,
                                                      {},
                                                      {}});
            } else {
                results.emplace_back(polisher::Sample{torch::cat(std::move(features_buffer)),
                                                      cat_vectors(positions_major_buffer),
                                                      cat_vectors(positions_minor_buffer),
                                                      torch::cat(std::move(depth_buffer)),
                                                      seq_id_buffer,
                                                      region_id_buffer,
                                                      {},
                                                      {}});
            }
            features_buffer = {std::move(sample.features)};
            positions_major_buffer = {std::move(sample.positions_major)};
            positions_minor_buffer = {std::move(sample.positions_minor)};
            depth_buffer = {std::move(sample.depth)};
            seq_id_buffer = sample.seq_id;
            region_id_buffer = sample.region_id;
        }
    }

    if (!features_buffer.empty()) {
        // The torch::cat is slow, so just move if there is nothing to concatenate.
        if (std::size(features_buffer) == 1) {
            results.emplace_back(polisher::Sample{std::move(features_buffer.front()),
                                                  std::move(positions_major_buffer.front()),
                                                  std::move(positions_minor_buffer.front()),
                                                  std::move(depth_buffer.front()),
                                                  seq_id_buffer,
                                                  region_id_buffer,
                                                  {},
                                                  {}});
        } else {
            results.emplace_back(polisher::Sample{torch::cat(std::move(features_buffer)),
                                                  cat_vectors(positions_major_buffer),
                                                  cat_vectors(positions_minor_buffer),
                                                  torch::cat(std::move(depth_buffer)),
                                                  seq_id_buffer,
                                                  region_id_buffer,
                                                  {},
                                                  {}});
        }
    }

    return results;
}

}  // namespace

EncoderCounts::EncoderCounts(const NormaliseType normalise_type,
                             const std::vector<std::string>& dtypes,
                             const std::string_view tag_name,
                             const int32_t tag_value,
                             const bool tag_keep_missing,
                             const std::string& read_group,
                             const int32_t min_mapq,
                             const bool symmetric_indels)
        : m_normalise_type{normalise_type},
          m_dtypes{dtypes},
          m_num_dtypes{static_cast<int32_t>(std::size(m_dtypes)) + 1},
          m_tag_name{tag_name},
          m_tag_value{tag_value},
          m_tag_keep_missing{tag_keep_missing},
          m_read_group{read_group},
          m_min_mapq{min_mapq},
          m_symmetric_indels{symmetric_indels},
          m_feature_indices{pileup_counts_norm_indices(dtypes, 1)} {}

Sample EncoderCounts::encode_region(BamFile& bam_file,
                                    const std::string& ref_name,
                                    const int64_t ref_start,
                                    const int64_t ref_end,
                                    const int32_t seq_id) const {
    constexpr size_t num_qstrat = 1;
    constexpr bool weibull_summation = false;

    // Compute the pileup.
    // NOTE: the `num_qstrat` is passed into the `num_homop` parameter as is done in `pileup_counts` in features.py.
    PileupData pileup = calculate_pileup(
            bam_file, ref_name, ref_start, ref_end, m_num_dtypes, m_dtypes, num_qstrat, m_tag_name,
            m_tag_value, m_tag_keep_missing, weibull_summation, m_read_group, m_min_mapq);

    // Create Torch tensors from the pileup.
    const size_t n_rows = std::size(PILEUP_BASES) * m_num_dtypes * num_qstrat;
    CountsResult pileup_tensors = plp_data_to_tensors(pileup, n_rows);

    if (!pileup_tensors.counts.numel()) {
        const std::string region =
                ref_name + ':' + std::to_string(ref_start + 1) + '-' + std::to_string(ref_end);
        spdlog::debug("Pileup-feature is zero-length for {} indicating no reads in this region.",
                      region);
        return {};
    }

    return counts_to_features(pileup_tensors, seq_id, m_symmetric_indels, m_feature_indices,
                              m_normalise_type);
}

torch::Tensor EncoderCounts::collate(std::vector<torch::Tensor> batch) const {
    return torch::stack(batch);
}

std::vector<polisher::Sample> EncoderCounts::merge_adjacent_samples(
        std::vector<Sample> samples) const {
    return merge_adjacent_samples_impl(std::move(samples));
}

}  // namespace dorado::polisher
