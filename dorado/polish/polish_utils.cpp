#include "polish_utils.h"

#include <ostream>

namespace dorado::polisher {

void print_tensor_shape(std::ostream& os, const torch::Tensor& tensor) {
    os << "[";
    for (size_t i = 0; i < tensor.sizes().size(); ++i) {
        os << tensor.size(i);
        if ((i + 1) < tensor.sizes().size()) {
            os << ", ";
        }
    }
    os << "]";
}

std::string tensor_shape_as_string(const torch::Tensor& tensor) {
    std::ostringstream oss;
    print_tensor_shape(oss, tensor);
    return oss.str();
}

}  // namespace dorado::polisher
