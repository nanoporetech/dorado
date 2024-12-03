#include "polish/architectures/encoder_read_alignment.h"

#include "polish/medaka_read_matrix.h"
#include "polish/polish_utils.h"
#include "utils/ssize.h"

#include <spdlog/spdlog.h>
#include <utils/timer_high_res.h>

#include <unordered_map>

namespace dorado::polisher {

namespace {

ReadAlignmentTensors read_matrix_data_to_tensors(ReadAlignmentData& data) {
    ReadAlignmentTensors result;

    // Allocate a tensor of the appropriate size directly for `result.counts` on the CPU
    result.counts = torch::empty({data.n_pos, data.buffer_reads, data.featlen}, torch::kInt8);

    // Copy the data from `data.matrix` into `result.counts`
    std::memcpy(result.counts.data_ptr<int8_t>(), data.matrix.data(),
                data.n_pos * data.buffer_reads * data.featlen * sizeof(int8_t));

    result.counts =
            result.counts.index({torch::indexing::Slice(),
                                 torch::indexing::Slice(0, static_cast<int64_t>(data.n_reads)),
                                 torch::indexing::Slice()});

    result.positions_major = std::move(data.major);
    result.positions_minor = std::move(data.minor);
    result.read_ids_left = std::move(data.read_ids_left);
    result.read_ids_right = std::move(data.read_ids_right);

    return result;
}

std::vector<Sample> merge_adjacent_samples_read_matrix(std::vector<Sample> samples) {
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

    const auto pad_reads = [](std::vector<torch::Tensor> chunks, int64_t target_depth) {
        // Determine the target depth if not provided
        if (target_depth == -1) {
            target_depth = 0;
            for (const auto& chunk : chunks) {
                target_depth = std::max(target_depth, chunk.size(1));
            }
        }

        // Pad each chunk to match the target depth
        std::vector<torch::Tensor> padded_chunks;
        for (auto& chunk : chunks) {
            int64_t pad_depth = target_depth - chunk.size(1);
            if (pad_depth > 0) {
                auto padding =
                        torch::zeros({chunk.size(0), pad_depth, chunk.size(2)}, chunk.options());
                spdlog::trace("[pad_reads] Padding depth: chunk.shape = {}, padding.shape = {}",
                              tensor_shape_as_string(chunk), tensor_shape_as_string(padding));
                auto concated = torch::cat({std::move(chunk), std::move(padding)}, 1);
                spdlog::trace("[pad_reads] Emplacing (1) chunk: concated.shape = {}",
                              tensor_shape_as_string(concated));
                padded_chunks.emplace_back(std::move(concated));
            } else {
                spdlog::trace("[pad_reads] Emplacing (2) chunk: chunk.shape = {}",
                              tensor_shape_as_string(chunk));
                padded_chunks.emplace_back(std::move(chunk));
            }
        }

        return padded_chunks;
    };

    const auto reorder_reads = [](std::vector<torch::Tensor> chunks,
                                  const std::vector<std::vector<std::string>>& read_ids_in,
                                  const std::vector<std::vector<std::string>>& read_ids_out) {
        spdlog::trace("[reorder_reads] Entered. chunks.size = {}", std::size(chunks));

        if (std::size(chunks) < 2) {
            return chunks;
        }

        std::vector<torch::Tensor> reordered_chunks{chunks[0]};

        std::vector<std::string> rids_out = read_ids_out[0];

        for (int64_t n = 1; n < dorado::ssize(chunks); ++n) {
            auto& chunk = chunks[n];
            // const auto& rids_out = read_ids_out[n - 1];
            const auto& rids_in = read_ids_in[n];

            spdlog::trace(
                    "[reorder_reads] n = {}, rids_out.size() = {}, rids_in.size() = {}, "
                    "chunk.shape = {}",
                    n, std::size(rids_out), std::size(rids_in), tensor_shape_as_string(chunk));

            // Create a lookup.
            std::unordered_map<std::string, int64_t> rids_in_map;
            for (int64_t i = 0; i < dorado::ssize(rids_in); ++i) {
                rids_in_map[rids_in[i]] = i;
            }

            // Find the indices of the out reads in the in reads.
            std::vector<int64_t> new_indices(std::size(rids_out), -1);
            for (int64_t i = 0; i < dorado::ssize(rids_out); ++i) {
                const auto& rid = rids_out[i];
                const auto it = rids_in_map.find(rid);
                new_indices[i] = (it != std::end(rids_in_map)) ? it->second : -1;
            }

            // Find missing indices.
            std::vector<int64_t> missing_out_indices;
            for (int64_t i = 0; i < dorado::ssize(new_indices); ++i) {
                if (new_indices[i] == -1) {
                    missing_out_indices.emplace_back(i);
                }
            }
            std::vector<int64_t> missing_in_indices;
            std::unordered_set<int64_t> new_indices_set(std::begin(new_indices),
                                                        std::end(new_indices));
            for (int64_t i = 0; i < dorado::ssize(rids_in); ++i) {
                if (new_indices_set.count(i) == 0) {
                    missing_in_indices.emplace_back(i);
                }
            }

            spdlog::trace(
                    "[reorder_reads] n = {}, missing_in_indices.size() = {}, "
                    "missing_out_indices.size() = {}",
                    n, std::size(missing_in_indices), std::size(missing_out_indices));

            spdlog::trace("[reorder_reads] n = {}, missing_in_indices: {}", n,
                          print_container_as_string(missing_in_indices, ", "));

            spdlog::trace("[reorder_reads] n = {}, missing_out_indices: {}", n,
                          print_container_as_string(missing_out_indices, ", "));

            // Fill out the gaps in the array with some of the extra indices.
            for (size_t i = 0;
                 i < std::min(std::size(missing_out_indices), std::size(missing_in_indices)); ++i) {
                new_indices[missing_out_indices[i]] = missing_in_indices[i];
            }

            // Add remaining missing in-indices.
            if (std::size(missing_in_indices) > std::size(missing_out_indices)) {
                new_indices.insert(std::end(new_indices),
                                   std::begin(missing_in_indices) + std::size(missing_out_indices),
                                   std::end(missing_in_indices));
            }

            spdlog::trace("[reorder_reads] n = {}, creating an empty reordered_chunk.", n);

            // Permute.
            auto reordered_chunk = torch::zeros(
                    {chunk.size(0),
                     static_cast<int64_t>(std::max(std::size(rids_out), std::size(rids_in))),
                     chunk.size(2)},
                    chunk.options());
            spdlog::trace(
                    "[reorder_reads] n = {}, before permute. reordered_chunk.shape = {}, "
                    "new_indices.size() = {}",
                    n, tensor_shape_as_string(reordered_chunk), std::size(new_indices));
            for (size_t i = 0; i < std::size(new_indices); ++i) {
                if (new_indices[i] == -1) {
                    continue;
                }
                reordered_chunk.index_put_({torch::indexing::Slice(), static_cast<int64_t>(i),
                                            torch::indexing::Slice()},
                                           chunk.index({torch::indexing::Slice(), new_indices[i],
                                                        torch::indexing::Slice()}));
            }

            reordered_chunks.emplace_back(std::move(reordered_chunk));

            spdlog::trace("[reorder_reads] n = {}, updating the previous out column.", n);

            // Update read_ids_out for the next chunk.
            if ((n + 1) < dorado::ssize(chunks)) {
                rids_out.clear();
                rids_out.resize(std::size(new_indices));
                // auto& tmp_next_rids_out = read_ids_out[n - 1];
                for (int64_t i = 0; i < dorado::ssize(new_indices); ++i) {
                    const int64_t idx = new_indices[i];
                    rids_out[i] = (idx == -1) ? ("__inserted_" + std::to_string(i))
                                              : read_ids_out[n][idx];
                }
            }

            spdlog::trace("[reorder_reads] n = {}, done.", n);
        }

        return reordered_chunks;
    };

    const auto merge_samples = [&samples, &cat_vectors, &pad_reads,
                                &reorder_reads](const std::vector<int64_t>& sample_ids) {
        // The torch::cat is slow, so just move if there is nothing to concatenate.
        if (std::empty(sample_ids)) {
            return Sample{};
        }
        if (std::size(sample_ids) == 1) {
            return std::move(samples[sample_ids.front()]);
        }

        std::vector<torch::Tensor> features;
        std::vector<std::vector<int64_t>> positions_major;
        std::vector<std::vector<int64_t>> positions_minor;
        std::vector<torch::Tensor> depth;
        std::vector<std::vector<std::string>> read_ids_left;
        std::vector<std::vector<std::string>> read_ids_right;

        const int32_t seq_id = samples[sample_ids.front()].seq_id;
        const int32_t region_id = samples[sample_ids.front()].region_id;
        bool is_last = false;

        // Make buffers.
        for (const int64_t id : sample_ids) {
            Sample& sample = samples[id];
            features.emplace_back(std::move(sample.features));
            depth.emplace_back(std::move(sample.depth));
            positions_major.emplace_back(std::move(sample.positions_major));
            positions_minor.emplace_back(std::move(sample.positions_minor));
            read_ids_left.emplace_back(std::move(sample.read_ids_left));
            read_ids_right.emplace_back(std::move(sample.read_ids_right));
            is_last = std::max(is_last, sample.is_last);
        }

        // NOTE: It appears that the read IDs are not supposed to be merged. After this stage it seems they are no longer needed.
        Sample ret{
                torch::cat(pad_reads(
                        reorder_reads(std::move(features), read_ids_left, read_ids_right), -1)),
                cat_vectors(positions_major),
                cat_vectors(positions_minor),
                torch::cat(std::move(depth)),
                seq_id,
                region_id,
                {},
                {},
                is_last,
        };

        return ret;
    };

    std::vector<int64_t> buffer_ids;
    int64_t last_end = -1;

    std::vector<polisher::Sample> results;

    for (int64_t i = 0; i < dorado::ssize(samples); ++i) {
        auto& sample = samples[i];

        if (std::empty(sample.positions_major)) {
            spdlog::trace("[merge_adjacent_samples_read_matrix] Empty sample: i = {}", i);
            continue;
        }

        const int64_t start = sample.start();

        const int64_t first_id = std::empty(buffer_ids) ? -1 : buffer_ids.front();

        if (std::empty(buffer_ids) ||
            ((sample.seq_id == samples[first_id].seq_id) &&
             (sample.region_id == samples[first_id].region_id) && ((start - last_end) == 0))) {
            // New or contiguous chunk.
            last_end = sample.end();
            buffer_ids.emplace_back(i);
            spdlog::trace("[merge_adjacent_samples_read_matrix] Emplacing to buffer_ids, i = {}",
                          i);

        } else {
            // Discontinuity found, finalize the current chunk.
            last_end = sample.end();
            results.emplace_back(merge_samples(buffer_ids));
            buffer_ids = {i};
            spdlog::trace(
                    "[merge_adjacent_samples_read_matrix] Merging samples in buffer, resetting the "
                    "buffer. i = {}",
                    i);
            spdlog::trace("[merge_adjacent_samples_read_matrix] Merged.");
        }
    }

    if (!buffer_ids.empty()) {
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

EncoderReadAlignment::EncoderReadAlignment(const std::vector<std::string>& dtypes,
                                           const std::string_view tag_name,
                                           const int32_t tag_value,
                                           const bool tag_keep_missing,
                                           const std::string_view read_group,
                                           const int32_t min_mapq,
                                           const int32_t max_reads,
                                           const bool row_per_read,
                                           const bool include_dwells,
                                           const bool include_haplotype)
        : m_num_dtypes{static_cast<int32_t>(std::size(dtypes)) + 1},
          m_dtypes{dtypes},
          m_tag_name{tag_name},
          m_tag_value{tag_value},
          m_tag_keep_missing{tag_keep_missing},
          m_read_group{read_group},
          m_min_mapq{min_mapq},
          m_max_reads{max_reads},
          m_row_per_read{row_per_read},
          m_include_dwells{include_dwells},
          m_include_haplotype{include_haplotype} {}

Sample EncoderReadAlignment::encode_region(BamFile& bam_file,
                                           const std::string& ref_name,
                                           const int64_t ref_start,
                                           const int64_t ref_end,
                                           const int32_t seq_id,
                                           const bool is_last) const {
    const char* read_group_ptr = std::empty(m_read_group) ? nullptr : m_read_group.c_str();

    // Compute the counts and data.
    ReadAlignmentData counts = calculate_read_alignment(
            bam_file, ref_name, ref_start, ref_end, m_num_dtypes, m_dtypes, m_tag_name, m_tag_value,
            m_tag_keep_missing, read_group_ptr, m_min_mapq, m_row_per_read, m_include_dwells,
            m_include_haplotype, m_max_reads);

    // Create Torch tensors from the pileup.
    ReadAlignmentTensors tensors = read_matrix_data_to_tensors(counts);

    if (!tensors.counts.numel()) {
        const std::string region =
                ref_name + ':' + std::to_string(ref_start + 1) + '-' + std::to_string(ref_end);
        spdlog::warn("Pileup-feature is zero-length for {} indicating no reads in this region.",
                     region);
        return {};
    }

    torch::Tensor depth = (tensors.counts.index({"...", 0}) != 0).sum(/*dim=*/1);

    Sample sample{std::move(tensors.counts),
                  std::move(tensors.positions_major),
                  std::move(tensors.positions_minor),
                  std::move(depth),
                  seq_id,
                  -1,
                  std::move(tensors.read_ids_left),
                  std::move(tensors.read_ids_right),
                  is_last};

    return sample;
}

torch::Tensor EncoderReadAlignment::collate(std::vector<torch::Tensor> batch) const {
    if (std::empty(batch)) {
        return {};
    }

    const int64_t batch_size = static_cast<int64_t>(std::size(batch));
    const auto feature_shape = batch.front().sizes();

    // Adjust negative values in features to 0.
    for (auto& data : batch) {
        data = torch::clamp_min(data, 0);
    }

    torch::Tensor features;

    // Process read-level features if the shape indicates a 3D tensor.
    if (std::size(feature_shape) == 3) {
        // spdlog::info("About to merge tensors.");

        const int64_t npos = feature_shape[0];
        const int64_t nfeats = feature_shape[2];

        // Compute max depth across samples.
        std::vector<int64_t> depths;
        depths.reserve(batch_size);
        for (const auto& data : batch) {
            depths.push_back(data.sizes()[1]);
        }
        const int64_t max_depth = *std::max_element(std::begin(depths), std::end(depths));

        // Initialize a zero-filled feature tensor.
        features = torch::zeros({batch_size, npos, max_depth, nfeats}, torch::kUInt8);

        // for (size_t i = 0; i < std::size(batch); ++i) {
        //     std::ostringstream oss;
        //     print_tensor_shape(oss, batch[i]);
        //     std::cerr << "[i = " << i << "] batch[i].shape = " << oss.str() << "\n";
        //     // spdlog::info("[i = {}] batch[i].shape = {}", i, oss.str());
        // }

        // Fill the tensor with sample data, padding as necessary.
        for (size_t i = 0; i < std::size(batch); ++i) {
            features.index_put_({static_cast<int64_t>(i), torch::indexing::Slice(),
                                 torch::indexing::Slice(0, depths[i]), torch::indexing::Slice()},
                                batch[i]);
        }
    }

    // // Calculate masks if requested.
    // std::unordered_map<std::string, torch::Tensor> masks;
    // if (calculate_masks) {
    //     // Assuming `medaka::common::get_sample_masks` is implemented elsewhere.
    //     std::vector<std::unordered_map<std::string, torch::Tensor>> masks_per_sample;
    //     for (const auto& sample : samples) {
    //         masks_per_sample.push_back(medaka::common::get_sample_masks(sample));
    //     }

    //     // Extract mask keys and stack the masks for each key.
    //     const auto& mask_keys = masks_per_sample[0];
    //     for (const auto& [key, _] : mask_keys) {
    //         std::vector<torch::Tensor> stacked_masks;
    //         for (const auto& sample_masks : masks_per_sample) {
    //             stacked_masks.push_back(sample_masks.at(key));
    //         }
    //         masks[key] = torch::stack(stacked_masks);
    //     }
    // }

    return features;
}

std::vector<polisher::Sample> EncoderReadAlignment::merge_adjacent_samples(
        std::vector<Sample> samples) const {
    return merge_adjacent_samples_read_matrix(std::move(samples));
}

}  // namespace dorado::polisher
