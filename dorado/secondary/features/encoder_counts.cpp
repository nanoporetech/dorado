#include "encoder_counts.h"

#include "medaka_counts.h"
#include "utils/ssize.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace dorado::secondary {

namespace {

/**
 * \brief Converts the vectors produced by the pileup function into proper tensors.
 */
CountsResult counts_data_to_tensors(PileupData& data, const size_t n_rows) {
    const size_t num_bytes = static_cast<size_t>(data.n_cols * n_rows * sizeof(int64_t));

    if (num_bytes == 0) {
        return {};
    }

    CountsResult result;

    // Allocate a tensor of the appropriate size on the CPU.
    result.counts = torch::empty({static_cast<long>(data.n_cols), static_cast<long>(n_rows)},
                                 torch::kInt64);

    assert(result.counts.data_ptr<int64_t>() != nullptr);

    // Copy the data from `data.matrix` into `result.counts`
    std::memcpy(result.counts.data_ptr<int64_t>(), std::data(data.matrix), num_bytes);

    result.positions_major = std::move(data.major);
    result.positions_minor = std::move(data.minor);

    return result;
}

/**
 * \brief Function to calculate feature vector normalization groups.
 * \param dtypes Vector of data type names (strings).
 * \param num_qstrat Qscore stratifications.
 * \return Lookup of the form: key = (dtype, strand) -> vector of indices
 */
FeatureIndicesType pileup_counts_norm_indices(const std::vector<std::string>& dtypes,
                                              const size_t num_qstrat) {
    // Create a map to store the indices.
    FeatureIndicesType indices;

    constexpr size_t featlen = std::size(PILEUP_BASES);

    // Iterate over each datatype.
    for (int64_t dti = 0; dti < dorado::ssize(dtypes); ++dti) {
        const std::string& dt = dtypes[dti];

        // Iterate over qscore stratification layers.
        for (int64_t qindex = 0; qindex < static_cast<int64_t>(num_qstrat); ++qindex) {
            // Iterate over the base codes (e.g., 'a', 'c', 'g', 't', etc.)
            for (int64_t base_i = 0; base_i < PILEUP_BASES_SIZE; ++base_i) {
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
 * \brief Normalizes the counts to produce the features for inference.
 */
secondary::Sample counts_to_features(CountsResult& pileup,
                                     const int32_t seq_id,
                                     const bool sym_indels,
                                     const FeatureIndicesType& feature_indices,
                                     const NormaliseType normalise_type) {
    // Avoid slow Torch operations as much as possible. The original Medaka code had this implemented
    // on a very high level with lots of redundancy in computation.

    // Get indices of minor positions.
    const int64_t num_rows = dorado::ssize(pileup.positions_major);
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

    // Compute the depth of each column.
    auto depth = pileup.counts.sum(1);

    // Set depth at minor positions to match the depth of the corresponding major position.
    auto depth_data = depth.data_ptr<int64_t>();
    for (size_t i = 0; i < std::size(minor_inds); ++i) {
        if (major_ind_at_minor_inds[i] != -1) {
            depth_data[minor_inds[i]] = depth_data[major_ind_at_minor_inds[i]];
        }
    }
    const auto depth_unsequezed = depth.unsqueeze(1).to(FeatureTensorType);

    // Symmetric indel handling.
    if (sym_indels) {
        const at::Tensor minor_inds_tensor = torch::tensor(minor_inds, torch::kInt64);
        const at::Tensor major_ind_at_minor_inds_tensor =
                torch::tensor(major_ind_at_minor_inds, torch::kInt64);

        for (const auto& [key, inds] : feature_indices) {
            // const std::string& data_type = kv.first.first;
            const bool is_rev = key.second;
            const at::Tensor inds_tensor = torch::tensor(inds, torch::dtype(torch::kInt64));

            const auto dt_depth =
                    pileup.counts.index({torch::indexing::Slice(), inds_tensor}).sum(1);

            // Define deletion index.
            const int64_t featlen_index = is_rev ? PILEUP_POS_DEL_REV : PILEUP_POS_DEL_FWD;
            const int64_t dtype_size = PILEUP_BASES_SIZE;

            // Find the deletion index
            for (const int64_t x : inds) {
                if ((x % dtype_size) == featlen_index) {
                    pileup.counts.index_put_({minor_inds_tensor, x},
                                             dt_depth.index({major_ind_at_minor_inds_tensor}) -
                                                     dt_depth.index({minor_inds_tensor}));
                }
            }
        }
    }

    at::Tensor feature_array;
    if (normalise_type == NormaliseType::TOTAL) {
        feature_array =
                pileup.counts / torch::max(depth_unsequezed, torch::ones_like(depth_unsequezed));

    } else if (normalise_type == NormaliseType::FWD_REV) {
        feature_array = torch::empty_like(pileup.counts, FeatureTensorType);
        const at::Tensor minor_inds_tensor = torch::tensor(minor_inds, torch::kInt64);
        const at::Tensor major_ind_at_minor_inds_tensor =
                torch::tensor(major_ind_at_minor_inds, torch::kInt64);
        for (const auto& kv : feature_indices) {
            const std::vector<int64_t>& inds = kv.second;
            const at::Tensor inds_tensor = torch::tensor(inds, torch::dtype(torch::kInt64));

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

    secondary::Sample sample{seq_id,
                             std::move(feature_array),
                             std::move(pileup.positions_major),
                             std::move(pileup.positions_minor),
                             std::move(depth),
                             {},
                             {}};

    return sample;
}

std::vector<secondary::Sample> merge_adjacent_samples_impl(std::vector<secondary::Sample> samples) {
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

    const auto merge_samples = [&samples, &cat_vectors](const std::vector<int64_t>& sample_ids) {
        if (std::empty(sample_ids)) {
            return secondary::Sample{};
        }

        // The torch::cat is slow, so just move if there is nothing to concatenate.
        if (std::size(sample_ids) == 1) {
            return std::move(samples[sample_ids.front()]);
        }

        std::vector<at::Tensor> features;
        std::vector<std::vector<int64_t>> positions_major;
        std::vector<std::vector<int64_t>> positions_minor;
        std::vector<at::Tensor> depth;

        const int32_t seq_id = samples[sample_ids.front()].seq_id;

        // Make buffers.
        for (const int64_t id : sample_ids) {
            secondary::Sample& sample = samples[id];
            features.emplace_back(std::move(sample.features));
            depth.emplace_back(std::move(sample.depth));
            positions_major.emplace_back(std::move(sample.positions_major));
            positions_minor.emplace_back(std::move(sample.positions_minor));
        }

        // NOTE: It appears that the read IDs are not supposed to be merged. After this stage it seems they are no longer needed.
        secondary::Sample ret{
                seq_id,
                torch::cat(std::move(features)),
                cat_vectors(positions_major),
                cat_vectors(positions_minor),
                torch::cat(std::move(depth)),
                {},
                {},
        };

        return ret;
    };

    std::vector<int64_t> buffer_ids;
    int64_t last_end = -1;

    std::vector<secondary::Sample> results;

    for (int64_t i = 0; i < dorado::ssize(samples); ++i) {
        // Non-const so that it can be moved by the lambdas.
        auto& sample = samples[i];

        if (std::empty(sample.positions_major)) {
            continue;
        }

        const int64_t start = sample.start();

        const int64_t first_id = std::empty(buffer_ids) ? -1 : buffer_ids.front();

        if (std::empty(buffer_ids) ||
            ((sample.seq_id == samples[first_id].seq_id) && ((start - last_end) == 0))) {
            // New or contiguous chunk.
            last_end = sample.end();
            buffer_ids.emplace_back(i);
            spdlog::trace("[merge_adjacent_samples_impl] Emplacing to buffer_ids, i = {}", i);

        } else {
            // Discontinuity found, finalize the current chunk.
            last_end = sample.end();
            results.emplace_back(merge_samples(buffer_ids));
            buffer_ids = {i};
            spdlog::trace(
                    "[merge_adjacent_samples_impl] Merging samples in buffer, resetting the "
                    "buffer. i = {}",
                    i);
            spdlog::trace("[merge_adjacent_samples_impl] Merged.");
        }
    }

    if (!std::empty(buffer_ids)) {
        spdlog::trace(
                "[merge_adjacent_samples_read_matrix] Final merging samples in buffer, resetting "
                "the buffer. buffer_ids.size() = {}",
                std::size(buffer_ids));
        results.emplace_back(merge_samples(buffer_ids));
        spdlog::trace("[merge_adjacent_samples_read_matrix] Merged.");
    }

    return results;
}

}  // namespace

EncoderCounts::EncoderCounts(const NormaliseType normalise_type,
                             const std::vector<std::string>& dtypes,
                             const std::string& tag_name,
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

secondary::Sample EncoderCounts::encode_region(secondary::BamFile& bam_file,
                                               const std::string& ref_name,
                                               const int64_t ref_start,
                                               const int64_t ref_end,
                                               const int32_t seq_id) const {
    constexpr size_t num_qstrat = 1;
    constexpr bool weibull_summation = false;

    // Compute the pileup.
    // NOTE: the `num_qstrat` is passed into the `num_homop` parameter as is done in `pileup_counts` in features.py.
    CountsResult pileup_tensors;
    try {
        PileupData pileup =
                calculate_pileup(bam_file, ref_name, ref_start, ref_end, m_num_dtypes, m_dtypes,
                                 num_qstrat, m_tag_name, m_tag_value, m_tag_keep_missing,
                                 weibull_summation, m_read_group, m_min_mapq);

        // Create Torch tensors from the pileup.
        const size_t n_rows = std::size(PILEUP_BASES) * m_num_dtypes * num_qstrat;
        pileup_tensors = counts_data_to_tensors(pileup, n_rows);

    } catch (const std::exception& e) {
        spdlog::warn(
                "[EncoderCounts] Could not encode region: {}:{}-{}! Caught error: '{}'. Returning "
                "empty.",
                ref_name, ref_start + 1, ref_end, e.what());
        return {};
    }

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

at::Tensor EncoderCounts::collate(std::vector<at::Tensor> batch) const {
    return torch::stack(batch);
}

std::vector<secondary::Sample> EncoderCounts::merge_adjacent_samples(
        std::vector<secondary::Sample> samples) const {
    return merge_adjacent_samples_impl(std::move(samples));
}

}  // namespace dorado::secondary
