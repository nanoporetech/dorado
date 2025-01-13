#include "sample.h"

#include "utils/ssize.h"

#include <ostream>

namespace dorado::polisher {

Sample slice_sample(const Sample& sample, const int64_t idx_start, const int64_t idx_end) {
    // Validate idx.
    const int64_t num_columns = dorado::ssize(sample.positions_major);
    if ((idx_start < 0) || (idx_start >= num_columns) || (idx_end >= idx_start) ||
        (idx_end > num_columns)) {
        throw std::out_of_range(
                "Index is out of range in slice_sample. idx_start = " + std::to_string(idx_start) +
                ", idx_end = " + std::to_string(idx_end) +
                ", num_columns = " + std::to_string(num_columns));
    }

    // Validate that the input data is sane.
    if ((dorado::ssize(sample.positions_minor) != num_columns) ||
        (sample.depth.size(0) != num_columns) || (sample.features.size(0) != num_columns)) {
        throw std::invalid_argument(
                "Input data dimensions are inconsistent. num_columns = " +
                std::to_string(num_columns) + ", sample.positions_minor.size = " +
                std::to_string(std::size(sample.positions_minor)) +
                ", sample.depth.size(0) = " + std::to_string(sample.depth.size(0)) +
                ", sample.features.size(0) = " + std::to_string(sample.features.size(0)));
    }

    // Create the sliced Sample.
    Sample sliced_sample;

    // Slice tensor data.
    sliced_sample.features = sample.features.index({at::indexing::Slice(idx_start, idx_end)});
    sliced_sample.depth = sample.depth.index({at::indexing::Slice(idx_start, idx_end)});

    // Slice vector data
    sliced_sample.positions_major =
            std::vector<int64_t>(std::begin(sample.positions_major) + idx_start,
                                 std::begin(sample.positions_major) + idx_end);
    sliced_sample.positions_minor =
            std::vector<int64_t>(std::begin(sample.positions_minor) + idx_start,
                                 std::begin(sample.positions_minor) + idx_end);

    // Copy meta-information.
    sliced_sample.seq_id = sample.seq_id;
    sliced_sample.region_id = sample.region_id;

    // Read IDs are not mandatory. Slice them only if available.
    if (!std::empty(sample.read_ids_left)) {
        const int64_t vec_len = dorado::ssize(sample.read_ids_left);
        if ((idx_start >= 0) && (idx_start < vec_len) && (idx_end >= 0) && (idx_end <= vec_len)) {
            sliced_sample.read_ids_left.insert(std::begin(sliced_sample.read_ids_left),
                                               std::begin(sample.read_ids_left) + idx_start,
                                               std::begin(sample.read_ids_left) + idx_end);
        }
    }
    if (!std::empty(sample.read_ids_right)) {
        const int64_t vec_len = dorado::ssize(sample.read_ids_right);
        if ((idx_start >= 0) && (idx_start < vec_len) && (idx_end >= 0) && (idx_end <= vec_len)) {
            sliced_sample.read_ids_right.insert(std::begin(sliced_sample.read_ids_right),
                                                std::begin(sample.read_ids_right) + idx_start,
                                                std::begin(sample.read_ids_right) + idx_end);
        }
    }

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
    os << ", depth.shape = " << tensor_shape_as_string(sample.depth);
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

}  // namespace dorado::polisher
