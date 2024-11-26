#pragma once

#include <torch/torch.h>

#include <iosfwd>
#include <sstream>
#include <string>

namespace dorado::polisher {

void print_tensor_shape(std::ostream& os, const torch::Tensor& tensor);

std::string tensor_shape_as_string(const torch::Tensor& tensor);

template <typename T>
void print_container(std::ostream& os, const T& data, const std::string& delimiter) {
    bool first = true;
    os << "[";
    for (const auto& val : data) {
        if (!first) {
            os << delimiter;
        }
        os << val;
        first = false;
    }
    os << "]";
}

template <typename T>
std::string print_container_as_string(const T& data, const std::string& delimiter) {
    std::ostringstream oss;
    print_container(oss, data, delimiter);
    return oss.str();
}

}  // namespace dorado::polisher
