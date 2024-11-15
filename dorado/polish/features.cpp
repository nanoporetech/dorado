#include "features.h"

#include "polish/medaka_counts.h"

#include <spdlog/spdlog.h>
#include <utils/timer_high_res.h>

namespace dorado::polisher {

namespace {

CountsResult plp_data_to_tensors(PileupData& data, const size_t n_rows) {
    CountsResult result;

    // Allocate a tensor of the appropriate size directly for `result.counts` on the CPU
    result.counts = torch::empty({static_cast<long>(data.n_cols()), static_cast<long>(n_rows)},
                                 torch::kInt64);

    // Copy the data from `data.matrix()` into `result.counts`
    std::memcpy(result.counts.data_ptr<int64_t>(), data.matrix().data(),
                data.n_cols() * n_rows * sizeof(int64_t));

    std::swap(result.positions_major, data.major());
    std::swap(result.positions_minor, data.minor());

    // std::cerr << "data.n_cols() = " << data.n_cols() << "\n";

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
CountsResult construct_pileup_counts(bam_fset& bam_set,
                                     const std::string& ref_name,
                                     const int32_t ref_start,
                                     const int32_t ref_end,
                                     size_t num_qstrat /*= 1*/,
                                     size_t num_dtypes /*= 1*/,
                                     const std::vector<std::string>& dtypes /*= {}*/,
                                     const std::string tag_name /*= {}*/,
                                     int tag_value /*= 0*/,
                                     bool keep_missing /*= false*/,
                                     bool weibull_summation /*= false*/,
                                     const char* read_group /*= NULL*/,
                                     const int min_mapq /*= 1*/) {
    // Compute the pileup.
    // NOTE: the `num_qstrat` is passed into the `num_homop` parameter as is done in `pileup_counts` in features.py.
    // NOTE 2: the from_blob expects non-const data, so can't define the pileup as const here.
    PileupData pileup = calculate_pileup(ref_name, ref_start, ref_end, bam_set, num_dtypes, dtypes,
                                         num_qstrat, tag_name, tag_value, keep_missing,
                                         weibull_summation, read_group, min_mapq);
    // Create Torch tensors from the pileup.
    const size_t n_rows = std::size(PILEUP_BASES) * num_dtypes * num_qstrat;
    CountsResult counts_result = plp_data_to_tensors(pileup, n_rows);

    return counts_result;

    // const auto find_gaps = [](const std::vector<int64_t>& positions,
    //                           int64_t threshold = 1) -> std::vector<int64_t> {
    //     std::vector<int64_t> ret;
    //     for (size_t i = 1; i < std::size(positions); ++i) {
    //         if ((positions[i] - positions[i - 1]) > threshold) {
    //             ret.emplace_back(i);
    //         }
    //     }
    //     return ret;
    // };

    // const auto split_on_discontinuities = [&find_gaps](std::vector<CountsResult>& pileups) {
    //     std::vector<CountsResult> split_results;

    //     // TODO: Reimplement this with iteration over data instead of so much tensor slicing.
    //     for (auto& data : pileups) {
    //         const std::vector<int64_t> gaps = find_gaps(data.positions_major);

    //         if (std::empty(gaps)) {
    //             split_results.emplace_back(std::move(data));
    //         } else {
    //             int64_t start = 0;
    //             for (const int64_t i : gaps) {
    //                 std::vector<int64_t> new_major_pos(data.positions_major.begin() + start,
    //                                                    data.positions_major.begin() + i);
    //                 std::vector<int64_t> new_minor_pos(data.positions_minor.begin() + start,
    //                                                    data.positions_minor.begin() + i);
    //                 split_results.emplace_back(CountsResult{data.counts.slice(0, start, i),
    //                                                         std::move(new_major_pos),
    //                                                         std::move(new_minor_pos)});
    //                 start = i;
    //             }
    //             if (start < static_cast<int64_t>(std::size(data.positions_major))) {
    //                 std::vector<int64_t> new_major_pos(data.positions_major.begin() + start,
    //                                                    data.positions_major.end());
    //                 std::vector<int64_t> new_minor_pos(data.positions_minor.begin() + start,
    //                                                    data.positions_minor.end());
    //                 split_results.emplace_back(CountsResult{data.counts.slice(0, start),
    //                                                         std::move(new_major_pos),
    //                                                         std::move(new_minor_pos)});
    //             }
    //         }
    //     }

    //     return split_results;
    // };

    // const auto cat_vectors = [](const std::vector<std::vector<int64_t>>& vecs) {
    //     size_t size = 0;
    //     for (const auto& vec : vecs) {
    //         size += std::size(vec);
    //     }
    //     std::vector<int64_t> ret;
    //     ret.reserve(size);
    //     for (const auto& vec : vecs) {
    //         ret.insert(ret.end(), vec.cbegin(), vec.cend());
    //     }
    //     return ret;
    // };

    // const auto merge_chunks = [&cat_vectors](std::vector<CountsResult>& pileups) {
    //     std::vector<torch::Tensor> counts_buffer;
    //     std::vector<std::vector<int64_t>> positions_major_buffer;
    //     std::vector<std::vector<int64_t>> positions_minor_buffer;
    //     int64_t last_major = -1;

    //     std::vector<CountsResult> results;

    //     for (auto& data : pileups) {
    //         if (std::empty(data.positions_major)) {
    //             continue;
    //         }
    //         const int64_t first_major = data.positions_major.front();
    //         if (counts_buffer.empty() || (first_major - last_major) == 1) {
    //             // New or contiguous chunk.
    //             last_major = data.positions_major.back();
    //             counts_buffer.emplace_back(std::move(data.counts));
    //             positions_major_buffer.emplace_back(std::move(data.positions_major));
    //             positions_minor_buffer.emplace_back(std::move(data.positions_minor));

    //         } else {
    //             // Discontinuity found, finalize the current chunk
    //             last_major = data.positions_major.back();

    //             // The torch::cat is slow, so just move if there is nothing to concatenate.
    //             if (std::size(counts_buffer) == 1) {
    //                 results.emplace_back(CountsResult{std::move(counts_buffer.front()),
    //                                                   std::move(positions_major_buffer.front()),
    //                                                   std::move(positions_minor_buffer.front())});
    //             } else {
    //                 results.emplace_back(CountsResult{
    //                         torch::cat(std::move(counts_buffer)),
    //                         cat_vectors(positions_major_buffer),
    //                         cat_vectors(positions_minor_buffer),
    //                 });
    //             }
    //             counts_buffer = {std::move(data.counts)};
    //             positions_major_buffer = {std::move(data.positions_major)};
    //             positions_minor_buffer = {std::move(data.positions_minor)};
    //         }
    //     }

    //     if (!counts_buffer.empty()) {
    //         // The torch::cat is slow, so just move if there is nothing to concatenate.
    //         if (std::size(counts_buffer) == 1) {
    //             results.emplace_back(CountsResult{std::move(counts_buffer.front()),
    //                                               std::move(positions_major_buffer.front()),
    //                                               std::move(positions_minor_buffer.front())});
    //         } else {
    //             results.emplace_back(CountsResult{
    //                     torch::cat(std::move(counts_buffer)),
    //                     cat_vectors(positions_major_buffer),
    //                     cat_vectors(positions_minor_buffer),
    //             });
    //         }
    //     }

    //     return results;
    // };

    // // First pass: split at discontinuities within each chunk.
    // std::vector<CountsResult> results;
    // results.emplace_back(std::move(counts_result));

    // results = split_on_discontinuities(results);

    // // Second pass: merge neighboring chunks if they have no distance between them.
    // results = merge_chunks(results);

    // return results;
}

Sample counts_to_features(CountsResult& pileup,
                          const int32_t seq_id,
                          const bool sym_indels,
                          const FeatureIndicesType& feature_indices,
                          const NormaliseType normalise_type) {
    // const int64_t start = pileup.positions_major.front();
    // const int64_t end = pileup.positions_major.back();

    // if ((start != ref_start) || ((end + 1) != ref_end)) {
    //     spdlog::warn(
    //             "Pileup counts do not span requested region, requested {}:{}-{}, received {}-{}, "
    //             "pileup.positions_major.size() = {}",
    //             ref_name, ref_start, ref_end, start, end, std::size(pileup.positions_major));
    // }

    // Avoid slow Torch operations as much as possible. The original Medaka code had this implemented
    // on a very high level with lots of redundancy in computation.
    const int64_t num_rows = static_cast<int64_t>(std::size(pileup.positions_major));
    std::vector<int64_t> minor_inds;
    std::vector<int64_t> major_pos_at_minor_inds;
    std::vector<int64_t> major_ind_at_minor_inds;
    minor_inds.reserve(num_rows);
    major_pos_at_minor_inds.reserve(num_rows);
    major_ind_at_minor_inds.reserve(num_rows);
    int32_t last_non_minor_index = -1;
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

    Sample sample{std::move(feature_array), std::move(pileup.positions_major),
                  std::move(pileup.positions_minor), std::move(depth), seq_id};

    return sample;
}

// /**
//  * \brief Takes an input sample and splits it bluntly if it has too many positions. This can happen when
//  *          there are many long insertions in an input window, and can easily cause out-of-memory issues on the GPU
//  *          if the sample is not split.
//  *          Splitting is implemented to match Medaka, where a simple sliding window is used to create smaller samples.
//  *          In case of a smalle trailing portion (smaller than chunk_len), a potentially large overlap is produced to
//  *          cover this region instead of just outputing the small chunk.
//  */
// std::vector<Sample> split_sample(const Sample& sample,
//                                  const int64_t chunk_len,
//                                  const int64_t chunk_overlap) {
//     if ((chunk_overlap < 0) || (chunk_overlap > chunk_len)) {
//         throw std::runtime_error(
//                 "Wrong chunk_overlap length. chunk_len = " + std::to_string(chunk_len) +
//                 ", chunk_overlap = " + std::to_string(chunk_overlap));
//     }

//     const int64_t sample_len = static_cast<int64_t>(std::size(sample.positions_major));

//     if (sample_len <= chunk_len) {
//         return {sample};
//     }

//     const auto create_chunk = [&](const int64_t start, const int64_t end) {
//         torch::Tensor new_features = sample.features.slice(0, start, start + chunk_len);
//         std::vector<int64_t> new_major(sample.positions_major.begin() + start,
//                                        sample.positions_major.begin() + end);
//         std::vector<int64_t> new_minor(sample.positions_minor.begin() + start,
//                                        sample.positions_minor.begin() + end);
//         torch::Tensor new_depth = sample.depth.slice(0, start, start + chunk_len);
//         return Sample{sample.ref_name,      std::move(new_features), std::move(new_major),
//                       std::move(new_minor), std::move(new_depth),    sample.region_start,
//                       sample.region_end,    sample.seq_id,           sample.window_id};
//     };

//     std::vector<Sample> result;

//     const int64_t step = chunk_len - chunk_overlap;

//     int64_t end = 0;
//     for (int64_t start = 0; start < (sample_len - chunk_len + 1); start += step) {
//         end = start + chunk_len;
//         // std::cerr << "[split_sample] sample_len = " << sample_len << ", start = " << start << ", end = " << end << "\n";
//         result.emplace_back(create_chunk(start, end));
//     }

//     // This will create a chunk with potentially large overlap.
//     if (end < sample_len) {
//         const int64_t start = sample_len - chunk_len;
//         end = sample_len;
//         // std::cerr << "[split_sample] sample_len = " << sample_len << ", start = " << start << ", end = " << end << "\n";
//         result.emplace_back(create_chunk(start, end));
//     }

//     return result;
// }

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
          m_feature_indices{pileup_counts_norm_indices(dtypes, 1)} {}

Sample CountsFeatureEncoder::encode_region(const std::string& ref_name,
                                           const int64_t ref_start,
                                           const int64_t ref_end,
                                           const int32_t seq_id) const {
    constexpr size_t num_qstrat = 1;
    constexpr bool weibull_summation = false;

    const int32_t num_dtypes = static_cast<int32_t>(std::size(m_dtypes)) + 1;
    const char* read_group_ptr = std::empty(m_read_group) ? nullptr : m_read_group.c_str();

    CountsResult pileup = construct_pileup_counts(
            *m_bam_set, ref_name, ref_start, ref_end, num_qstrat, num_dtypes, m_dtypes, m_tag_name,
            m_tag_value, m_tag_keep_missing, weibull_summation, read_group_ptr, m_min_mapq);

    if (!pileup.counts.numel()) {
        const std::string region =
                ref_name + ':' + std::to_string(ref_start + 1) + '-' + std::to_string(ref_end);
        spdlog::warn("Pileup-feature is zero-length for {} indicating no reads in this region.",
                     region);
        return {};
    }

    return counts_to_features(pileup, seq_id, m_symmetric_indels, m_feature_indices,
                              m_normalise_type);
}

std::vector<ConsensusResult> CountsFeatureDecoder::decode_bases(const torch::Tensor& logits) {
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

    const torch::Tensor probs = torch::gather(logits, -1, indices.unsqueeze(-1)).squeeze(-1);

    // std::cerr << "probs: " << probs << "\n";

    for (int64_t sample_id = 0; sample_id < indices.size(0); ++sample_id) {
        std::string& quals = results[sample_id].quals;
        quals.clear();

        const auto phred_scores =
                (-10.0 * torch::log10(1.0 - probs[sample_id])).clamp(0, 40).to(torch::kUInt8) + 33;

        quals.resize(phred_scores.size(0), '!');
        for (int64_t j = 0; j < phred_scores.size(0); ++j) {
            quals[j] = static_cast<char>(phred_scores[j].item<uint8_t>());
        }
    }

    return results;
}

}  // namespace dorado::polisher
