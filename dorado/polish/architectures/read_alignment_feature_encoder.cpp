#include "polish/architectures/read_alignment_feature_encoder.h"

#include "polish/medaka_read_matrix.h"
#include "polish/polish_utils.h"

#include <spdlog/spdlog.h>
#include <utils/timer_high_res.h>

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

}  // namespace

ReadAlignmentFeatureEncoder::ReadAlignmentFeatureEncoder(const int32_t min_mapq)
        : m_min_mapq{min_mapq} {}

ReadAlignmentFeatureEncoder::ReadAlignmentFeatureEncoder(const std::vector<std::string>& dtypes,
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

Sample ReadAlignmentFeatureEncoder::encode_region(BamFile& bam_file,
                                                  const std::string& ref_name,
                                                  const int64_t ref_start,
                                                  const int64_t ref_end,
                                                  const int32_t seq_id) const {
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

    Sample sample{std::move(tensors.counts), std::move(tensors.positions_major),
                  std::move(tensors.positions_minor), std::move(depth), seq_id};

    return sample;
}

torch::Tensor ReadAlignmentFeatureEncoder::collate(std::vector<torch::Tensor> batch) const {
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
        spdlog::info("About to merge tensors.");

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

std::vector<ConsensusResult> ReadAlignmentFeatureDecoder::decode_bases(
        const torch::Tensor& logits) const {
    const auto indices = logits.argmax(-1);  // Shape becomes [N, L]

    std::vector<ConsensusResult> results(indices.size(0));

    for (int64_t sample_id = 0; sample_id < indices.size(0); ++sample_id) {
        const auto& positions = indices[sample_id];

        std::string& seq = results[sample_id].seq;
        seq.resize(positions.size(0), '*');

        for (int64_t j = 0; j < positions.size(0); ++j) {
            const int64_t class_index = positions[j].item<int64_t>();
            assert(class_index < static_cast<int64_t>(std::size(m_label_scheme)));
            seq[j] = m_label_scheme[class_index];
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
