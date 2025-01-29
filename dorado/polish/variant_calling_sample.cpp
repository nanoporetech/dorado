#include "variant_calling_sample.h"

#include "utils/ssize.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace dorado::polisher {

void VariantCallingSample::validate() const {
    if (seq_id < 0) {
        std::ostringstream oss;
        oss << "VariantCallingSample::seq_id < 0. seq_id = " << seq_id;
        throw std::runtime_error(oss.str());
    }

    // Validate lengths.
    if (std::size(positions_major) != std::size(positions_minor)) {
        std::ostringstream oss;
        oss << "VariantCallingSample::positions_major and positions_minor are not of same size. "
               "positions_major.size = "
            << std::size(positions_major)
            << ", positions_minor.size = " << std::size(positions_minor);
        throw std::runtime_error(oss.str());
    }

    if (!logits.defined()) {
        throw std::runtime_error("VariantCallingSample::logits tensor is not defined!");
    }

    const int64_t num_columns = dorado::ssize(positions_major);

    if (logits.size(0) != num_columns) {
        std::ostringstream oss;
        oss << "VariantCallingSample::logits is of incorrect size. logits.size = " << logits.size(0)
            << ", num_columns = " << num_columns;
        throw std::runtime_error(oss.str());
    }
}

int64_t VariantCallingSample::start() const {
    return (std::empty(positions_major) ? -1 : (positions_major.front()));
}

int64_t VariantCallingSample::end() const {
    return (std::empty(positions_major) ? -1 : (positions_major.back() + 1));
}

std::ostream& operator<<(std::ostream& os, const VariantCallingSample& vc_sample) {
    // Make sure that vectors are of the same length.
    vc_sample.validate();

    // Print scalar info.
    os << "seq_id = " << vc_sample.seq_id << ", positions = " << vc_sample.start() << " - "
       << vc_sample.end() << " , dist = " << (vc_sample.end() - vc_sample.start())
       << ", values = [";

    // Print first the beginning and end of the positions vectors.
    constexpr int64_t START = 0;
    const int64_t len = dorado::ssize(vc_sample.positions_major);
    for (int64_t k = START; k < std::min<int64_t>(START + 3, len); ++k) {
        os << "(" << vc_sample.positions_major[k] << ", " << vc_sample.positions_minor[k] << ") ";
        os.flush();
    }
    os << " ...";
    os.flush();
    const int64_t end = len;
    for (int64_t k = std::max<int64_t>(START + 3, end - 3); k < end; ++k) {
        os << " (" << vc_sample.positions_major[k] << ", " << vc_sample.positions_minor[k] << ")";
        os.flush();
    }
    os << "], size = " << std::size(vc_sample.positions_major);
    os.flush();

    return os;
}

}  // namespace dorado::polisher