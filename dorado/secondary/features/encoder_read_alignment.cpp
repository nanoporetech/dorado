#include "encoder_read_alignment.h"

#include "encoder_utils.h"
#include "medaka_read_matrix.h"
#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"
#include "utils/ssize.h"
#include "utils/timer_high_res.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::secondary {

namespace {

/**
 * \brief Converts the data into proper tensors.
 */
ReadAlignmentTensors read_matrix_data_to_tensors(ReadAlignmentData& data) {
    const size_t num_bytes =
            static_cast<size_t>(data.n_pos * data.buffer_reads * data.featlen * sizeof(int8_t));

    if (num_bytes == 0) {
        return {};
    }

    ReadAlignmentTensors result;

    // Allocate a tensor of the appropriate size directly for `result.counts` on the CPU
    result.counts = torch::empty({data.n_pos, data.buffer_reads, data.featlen}, torch::kInt8);

    assert(result.counts.data_ptr<int8_t>() != nullptr);

    // Copy the data from `data.matrix` into `result.counts`
    std::memcpy(result.counts.data_ptr<int8_t>(), std::data(data.matrix), num_bytes);

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

    const auto pad_reads = [](std::vector<at::Tensor> chunks, int64_t target_depth) {
        // Determine the target depth if not provided
        if (target_depth < 0) {
            target_depth = 0;
            for (const auto& chunk : chunks) {
                target_depth = std::max(target_depth, chunk.size(1));
            }
        }

        // Pad each chunk to match the target depth
        std::vector<at::Tensor> padded_chunks;
        for (auto& chunk : chunks) {
            const int64_t pad_depth = target_depth - chunk.size(1);
            if (pad_depth > 0) {
                auto padding =
                        torch::zeros({chunk.size(0), pad_depth, chunk.size(2)}, chunk.options());

                LOG_TRACE("[pad_reads] Padding depth: chunk.shape = {}, padding.shape = {}",
                          utils::tensor_shape_as_string(chunk),
                          utils::tensor_shape_as_string(padding));

                auto concated = torch::cat({std::move(chunk), std::move(padding)}, 1);

                LOG_TRACE("[pad_reads] Emplacing (1) chunk: concated.shape = {}",
                          utils::tensor_shape_as_string(concated));

                padded_chunks.emplace_back(std::move(concated));
            } else {
                LOG_TRACE("[pad_reads] Emplacing (2) chunk: chunk.shape = {}",
                          utils::tensor_shape_as_string(chunk));

                padded_chunks.emplace_back(std::move(chunk));
            }
        }

        return padded_chunks;
    };

    /**
     * \brief This lambda reorders reads in the chunk tensors, because some rows (reads) may be missing
     *          between neighboring samples (e.g. some reads end/begin).
     */
    const auto reorder_reads = [](std::vector<at::Tensor> chunks,
                                  const std::vector<std::vector<std::string>>& read_ids_in,
                                  const std::vector<std::vector<std::string>>& read_ids_out) {
        LOG_TRACE("[reorder_reads] Entered. chunks.size = {}", std::size(chunks));

        if (std::size(chunks) < 2) {
            return chunks;
        }

        std::vector<at::Tensor> reordered_chunks{chunks[0]};

        std::vector<std::string> prev_rids_out = read_ids_out[0];

        for (int64_t n = 1; n < dorado::ssize(chunks); ++n) {
            LOG_TRACE("[reorder_reads] Reordering chunk n = {}", n);

            auto [reordered_chunk, next_rids_out] =
                    reorder_chunk(chunks[n], prev_rids_out, read_ids_in[n], read_ids_out[n]);

            reordered_chunks.emplace_back(std::move(reordered_chunk));

            prev_rids_out = std::move(next_rids_out);

            LOG_TRACE("[reorder_reads] n = {}, updating the previous out column.", n);

            LOG_TRACE("[reorder_reads] n = {}, done.", n);
        }

        return reordered_chunks;
    };

    const auto merge_samples = [&samples, &cat_vectors, &pad_reads,
                                &reorder_reads](const std::vector<int64_t>& sample_ids) {
        // The torch::cat is slow, so just move if there is nothing to concatenate.
        if (std::empty(sample_ids)) {
            return secondary::Sample{};
        }
        if (std::size(sample_ids) == 1) {
            return std::move(samples[sample_ids.front()]);
        }

        std::vector<at::Tensor> features;
        std::vector<std::vector<int64_t>> positions_major;
        std::vector<std::vector<int64_t>> positions_minor;
        std::vector<at::Tensor> depth;
        std::vector<std::vector<std::string>> read_ids_left;
        std::vector<std::vector<std::string>> read_ids_right;

        const int32_t seq_id = samples[sample_ids.front()].seq_id;

        // Make buffers.
        for (const int64_t id : sample_ids) {
            secondary::Sample& sample = samples[id];
            features.emplace_back(std::move(sample.features));
            depth.emplace_back(std::move(sample.depth));
            positions_major.emplace_back(std::move(sample.positions_major));
            positions_minor.emplace_back(std::move(sample.positions_minor));
            read_ids_left.emplace_back(std::move(sample.read_ids_left));
            read_ids_right.emplace_back(std::move(sample.read_ids_right));
        }

        // NOTE: It appears that the read IDs are not supposed to be merged. After this stage it seems they are no longer needed.
        secondary::Sample ret{
                seq_id,
                torch::cat(pad_reads(
                        reorder_reads(std::move(features), read_ids_left, read_ids_right), -1)),
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
        auto& sample = samples[i];

        if (std::empty(sample.positions_major)) {
            LOG_TRACE("[merge_adjacent_samples_read_matrix] Empty sample: i = {}", i);
            continue;
        }

        const int64_t start = sample.start();

        const int64_t first_id = std::empty(buffer_ids) ? -1 : buffer_ids.front();

        if (std::empty(buffer_ids) ||
            ((sample.seq_id == samples[first_id].seq_id) && ((start - last_end) == 0))) {
            // New or contiguous chunk.
            last_end = sample.end();
            buffer_ids.emplace_back(i);
            LOG_TRACE("[merge_adjacent_samples_read_matrix] Emplacing to buffer_ids, i = {}", i);

        } else {
            // Discontinuity found, finalize the current chunk.
            last_end = sample.end();
            results.emplace_back(merge_samples(buffer_ids));
            buffer_ids = std::vector{i};
            LOG_TRACE(
                    "[merge_adjacent_samples_read_matrix] Merging samples in buffer, resetting the "
                    "buffer. i = {}",
                    i);
            LOG_TRACE("[merge_adjacent_samples_read_matrix] Merged.");
        }
    }

    if (!std::empty(buffer_ids)) {
        LOG_TRACE(
                "[merge_adjacent_samples_read_matrix] Final merging samples in buffer, resetting "
                "the buffer. buffer_ids.size() = {}",
                std::size(buffer_ids));
        results.emplace_back(merge_samples(buffer_ids));
        LOG_TRACE("[merge_adjacent_samples_read_matrix] Merged.");
    }

    return results;
}

}  // namespace

EncoderReadAlignment::EncoderReadAlignment(const std::filesystem::path& in_ref_fn,
                                           const std::filesystem::path& in_bam_aln_fn,
                                           const std::vector<std::string>& dtypes,
                                           const std::string& tag_name,
                                           const int32_t tag_value,
                                           const bool tag_keep_missing,
                                           const std::string& read_group,
                                           const int32_t min_mapq,
                                           const int32_t max_reads,
                                           const double min_snp_accuracy,
                                           const bool row_per_read,
                                           const bool include_dwells,
                                           const bool clip_to_zero,
                                           const bool right_align_insertions,
                                           const bool include_haplotype_column,
                                           const HaplotagSource hap_source,
                                           const std::optional<std::filesystem::path>& phasing_bin,
                                           const bool include_snp_qv_column)
        : m_fastx_reader{in_ref_fn},
          m_bam_file{secondary::BamFile(in_bam_aln_fn)},
          m_dtypes{dtypes},
          m_num_dtypes{static_cast<int32_t>(std::size(dtypes)) + 1},
          m_tag_name{tag_name},
          m_tag_value{tag_value},
          m_tag_keep_missing{tag_keep_missing},
          m_read_group{read_group},
          m_min_mapq{min_mapq},
          m_max_reads{max_reads},
          m_min_snp_accuracy{min_snp_accuracy},
          m_row_per_read{row_per_read},
          m_include_dwells{include_dwells},
          m_include_haplotype_column{include_haplotype_column},
          m_include_snp_qv_column{include_snp_qv_column},
          m_hap_source{hap_source},
          m_clip_to_zero{clip_to_zero},
          m_right_align_insertions{right_align_insertions},
          m_phasing_bin{phasing_bin},
          m_feature_column_map{produce_feature_column_map(include_dwells,
                                                          include_haplotype_column,
                                                          include_snp_qv_column,
                                                          (m_num_dtypes > 1))} {}

secondary::Sample EncoderReadAlignment::encode_region(const std::string& ref_name,
                                                      const int64_t ref_start,
                                                      const int64_t ref_end,
                                                      const int32_t seq_id) {
    // Compute the counts and data.
    ReadAlignmentTensors tensors;
    try {
        std::unique_lock<std::mutex> lock(m_mtx);

        const std::string phasing_bin_fn_str =
                (m_phasing_bin) ? m_phasing_bin->string() : std::string();

        ReadAlignmentData counts = calculate_read_alignment(
                m_bam_file, m_fastx_reader.get_raw_faidx_ptr(), ref_name, ref_start, ref_end,
                m_num_dtypes, m_dtypes, m_tag_name, m_tag_value, m_tag_keep_missing, m_read_group,
                m_min_mapq, m_row_per_read, m_include_dwells, m_include_haplotype_column,
                m_include_snp_qv_column, m_hap_source, phasing_bin_fn_str, m_max_reads,
                m_right_align_insertions, m_min_snp_accuracy);

        // Create Torch tensors from the pileup.
        tensors = read_matrix_data_to_tensors(counts);
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "[EncoderReadAlignment] Could not encode region: " << ref_name << ':'
            << (ref_start + 1) << '-' << ref_end << "! Original message: '" << e.what() << "'";
        throw std::runtime_error{oss.str()};
    }

    if (!tensors.counts.numel()) {
        const std::string region =
                ref_name + ':' + std::to_string(ref_start + 1) + '-' + std::to_string(ref_end);
        spdlog::debug("Pileup-feature is zero-length for {} indicating no reads in this region.",
                      region);
        return {};
    }

    at::Tensor depth = (tensors.counts.index({"...", 0}) != 0).sum(/*dim=*/1);

    secondary::Sample sample{
            .seq_id = seq_id,
            .features = std::move(tensors.counts),
            .positions_major = std::move(tensors.positions_major),
            .positions_minor = std::move(tensors.positions_minor),
            .depth = std::move(depth),
            .read_ids_left = std::move(tensors.read_ids_left),
            .read_ids_right = std::move(tensors.read_ids_right),
    };

    if (m_clip_to_zero) {
        sample.features = torch::clamp_min(sample.features, 0);
    }

    return sample;
}

at::Tensor EncoderReadAlignment::collate(std::vector<at::Tensor> batch) const {
    if (std::empty(batch)) {
        return {};
    }

    const int64_t batch_size = static_cast<int64_t>(std::size(batch));
    const auto feature_shape = batch.front().sizes();

    // Adjust negative values in features to 0.
    for (auto& data : batch) {
        data = torch::clamp_min(data, 0);
    }

    at::Tensor features;

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

        // Fill the tensor with sample data, padding as necessary.
        for (size_t i = 0; i < std::size(batch); ++i) {
            features.index_put_({static_cast<int64_t>(i), torch::indexing::Slice(),
                                 torch::indexing::Slice(0, depths[i]), torch::indexing::Slice()},
                                batch[i]);
        }
    } else {
        spdlog::warn(
                "Unsupported feature shape when collating samples. Shape: [{}], expected 3 "
                "dimensions. Returning an uninitialized features tensor.",
                utils::tensor_shape_as_string(batch.front()));
    }

    return features;
}

std::vector<secondary::Sample> EncoderReadAlignment::merge_adjacent_samples(
        std::vector<secondary::Sample> samples) const {
    return merge_adjacent_samples_impl(std::move(samples));
}

FeatureColumnMap EncoderReadAlignment::get_feature_column_map() const {
    return m_feature_column_map;
}

FeatureColumnMap EncoderReadAlignment::produce_feature_column_map(
        const bool include_dwells,
        const bool include_haplotype_column,
        const bool include_snp_qv_column,
        const bool include_dtypes) {
    // Fixed columns.
    FeatureColumnMap feature_column_map = {
            {FeatureColumns::BASE, 0},
            {FeatureColumns::QUAL, 1},
            {FeatureColumns::STRAND, 2},
            {FeatureColumns::MAPQ, 3},
    };

    // Optional columns.
    if (include_dwells) {
        const int32_t id = static_cast<int32_t>(std::ssize(feature_column_map));
        feature_column_map[FeatureColumns::DWELL] = id;
    }
    if (include_haplotype_column) {
        const int32_t id = static_cast<int32_t>(std::ssize(feature_column_map));
        feature_column_map[FeatureColumns::HAPLOTAG] = id;
    }
    if (include_snp_qv_column) {
        const int32_t id = static_cast<int32_t>(std::ssize(feature_column_map));
        feature_column_map[FeatureColumns::SNP_QV] = id;
    }
    if (include_dtypes) {
        const int32_t id = static_cast<int32_t>(std::ssize(feature_column_map));
        feature_column_map[FeatureColumns::DTYPE] = id;
    }

    return feature_column_map;
}

}  // namespace dorado::secondary
