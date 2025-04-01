#include "sample.h"

#include "torch_utils/tensor_utils.h"
#include "utils/ssize.h"

#include <torch/types.h>

#include <ostream>

namespace dorado::secondary {

void Sample::validate() const {
    const int64_t num_columns = dorado::ssize(positions_major);

    if (!features.defined()) {
        throw std::runtime_error("Sample::features tensor is not defined!");
    }

    if (!depth.defined()) {
        throw std::runtime_error("Sample::depth tensor is not defined!");
    }

    // Validate that the input data is sane.
    if ((dorado::ssize(positions_minor) != num_columns) || (depth.size(0) != num_columns) ||
        (features.size(0) != num_columns)) {
        throw std::runtime_error(
                "Sample data dimensions are inconsistent. positions_major.size = " +
                std::to_string(std::size(positions_major)) +
                ", positions_minor.size = " + std::to_string(std::size(positions_minor)) +
                ", depth.size(0) = " + std::to_string(depth.size(0)) +
                ", features.size(0) = " + std::to_string(features.size(0)));
    }
}

Sample slice_sample(const Sample& sample, const int64_t idx_start, const int64_t idx_end) {
    sample.validate();

    // Validate idx.
    const int64_t num_columns = dorado::ssize(sample.positions_major);
    if ((idx_start < 0) || (idx_start >= num_columns) || (idx_start >= idx_end) ||
        (idx_end > num_columns)) {
        throw std::out_of_range(
                "Index is out of range in slice_sample. idx_start = " + std::to_string(idx_start) +
                ", idx_end = " + std::to_string(idx_end) +
                ", num_columns = " + std::to_string(num_columns));
    }

    // Create the sliced Sample.
    Sample sliced_sample;

    // Features.
    sliced_sample.features =
            sample.features.index({at::indexing::Slice(idx_start, idx_end)}).clone();

    // Depth.
    sliced_sample.depth = sample.depth.index({at::indexing::Slice(idx_start, idx_end)}).clone();

    // Major positions.
    sliced_sample.positions_major =
            std::vector<int64_t>(std::begin(sample.positions_major) + idx_start,
                                 std::begin(sample.positions_major) + idx_end);

    // Minor positions.
    sliced_sample.positions_minor =
            std::vector<int64_t>(std::begin(sample.positions_minor) + idx_start,
                                 std::begin(sample.positions_minor) + idx_end);

    // Meta information.
    sliced_sample.seq_id = sample.seq_id;

    // Not needed, but stating for clarity. Slicing will not produce these.
    sliced_sample.read_ids_left.clear();
    sliced_sample.read_ids_right.clear();

    return sliced_sample;
}

void debug_print_sample(std::ostream& os,
                        const Sample& sample,
                        int64_t start /*= 0*/,
                        int64_t end /*= -1 */,
                        bool debug /*= false */) {
    const int64_t len = static_cast<int64_t>(std::size(sample.positions_major));
    start = std::max<int64_t>(0, start);
    end = (end <= 0) ? len : end;

    os << "sample.positions = " << sample.start() << " - " << sample.end()
       << " , dist = " << (sample.end() - sample.start()) << " , tensor = [";
    os.flush();
    for (int64_t k = start; k < std::min<int64_t>(start + 3, len); ++k) {
        os << "(" << sample.positions_major[k] << ", " << sample.positions_minor[k] << ") ";
        os.flush();
    }
    os << " ...";
    os.flush();
    for (int64_t k = std::max<int64_t>(0, end - 3); k < end; ++k) {
        os << " (" << sample.positions_major[k] << ", " << sample.positions_minor[k] << ")";
        os.flush();
    }
    os << "], size = " << std::size(sample.positions_major);
    os << ", depth.shape = " << utils::tensor_shape_as_string(sample.depth);
    os.flush();

    if (debug) {
        const auto depth = sample.depth.slice(/*dim=*/0, /*start=*/0);
        for (int64_t k = 0; k < len; ++k) {
            os << "[k = " << k << "] pos = (" << sample.positions_major[k] << ", "
               << sample.positions_minor[k] << "), depth = " << depth[k].item<float>() << '\n';
            os.flush();
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Sample& sample) {
    debug_print_sample(os, sample, 0, -1, false);
    return os;
}

void merge_adjacent_samples_in_place(Sample& lh, const Sample& rh) {
    if (lh.seq_id != rh.seq_id) {
        std::ostringstream oss;
        oss << "Cannot merge samples. Different seq_id. lh = " << lh << ", rh = " << rh;
        throw std::runtime_error(oss.str());
    }
    if (lh.end() != (rh.start() + 1)) {
        std::ostringstream oss;
        oss << "Cannot merge samples, coordinates are not adjacent. lh = " << lh << ", rh = " << rh;
        throw std::runtime_error(oss.str());
    }

    const size_t width = std::size(lh.positions_major);

    // Merge the tensors.
    lh.features = torch::cat({std::move(lh.features), rh.features});
    lh.depth = torch::cat({std::move(lh.depth), rh.depth});

    // Insert positions vectors.
    lh.positions_major.reserve(width + std::size(rh.positions_major));
    lh.positions_major.insert(std::end(lh.positions_major), std::begin(rh.positions_major),
                              std::end(rh.positions_major));
    lh.positions_minor.reserve(width + std::size(rh.positions_minor));
    lh.positions_minor.insert(std::end(lh.positions_minor), std::begin(rh.positions_minor),
                              std::end(rh.positions_minor));

    // Invalidate read IDs.
    lh.read_ids_left.clear();
    lh.read_ids_right.clear();
}

}  // namespace dorado::secondary
