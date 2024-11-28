#pragma once

#include "polish/interval.h"

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

template <typename T, typename F>
std::vector<Interval> create_batches(const T& data,
                                     const int64_t batch_size,
                                     const F& functor_data_size) {
    std::vector<polisher::Interval> ret;
    polisher::Interval interval{0, 0};
    int64_t sum = 0;
    for (const auto& val : data) {
        const int64_t s = functor_data_size(val);
        sum += s;
        ++interval.end;
        if (sum >= batch_size) {
            ret.emplace_back(interval);
            interval.start = interval.end;
            sum = 0;
        }
    }
    if (interval.end > interval.start) {
        ret.emplace_back(interval);
    }
    return ret;
}

}  // namespace dorado::polisher
