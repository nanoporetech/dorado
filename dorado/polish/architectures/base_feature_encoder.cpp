#include "polish/architectures/base_feature_encoder.h"

#include <spdlog/spdlog.h>

namespace dorado::polisher {

std::vector<polisher::Sample> BaseFeatureEncoder::merge_adjacent_samples(
        std::vector<polisher::Sample> samples) const {
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
                                                      seq_id_buffer, region_id_buffer});
            } else {
                results.emplace_back(polisher::Sample{
                        torch::cat(std::move(features_buffer)), cat_vectors(positions_major_buffer),
                        cat_vectors(positions_minor_buffer), torch::cat(std::move(depth_buffer)),
                        seq_id_buffer, region_id_buffer});
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
            results.emplace_back(polisher::Sample{
                    std::move(features_buffer.front()), std::move(positions_major_buffer.front()),
                    std::move(positions_minor_buffer.front()), std::move(depth_buffer.front()),
                    seq_id_buffer, region_id_buffer});
        } else {
            results.emplace_back(polisher::Sample{
                    torch::cat(std::move(features_buffer)), cat_vectors(positions_major_buffer),
                    cat_vectors(positions_minor_buffer), torch::cat(std::move(depth_buffer)),
                    seq_id_buffer, region_id_buffer});
        }
    }

    return results;
}

}  // namespace dorado::polisher
