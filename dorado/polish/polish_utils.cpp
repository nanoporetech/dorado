#include "polish_utils.h"

#include <htslib/faidx.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <ostream>

namespace dorado::polisher {

void print_tensor_shape(std::ostream& os, const at::Tensor& tensor, const std::string& delimiter) {
    for (size_t i = 0; i < tensor.sizes().size(); ++i) {
        os << tensor.size(i);
        if ((i + 1) < tensor.sizes().size()) {
            os << delimiter;
        }
    }
}

std::string tensor_shape_as_string(const at::Tensor& tensor) {
    std::ostringstream oss;
    print_tensor_shape(oss, tensor, ", ");
    return oss.str();
}

std::vector<int32_t> parse_int32_vector(const std::string& input) {
    if (std::empty(input)) {
        return {};
    }
    if ((std::size(input) < 2) || (input.front() != '[') || (input.back() != ']')) {
        throw std::runtime_error("Input string must start with '[' and end with ']'.");
    }

    // Remove the brackets and trim the string
    std::string trimmed = input.substr(1, std::size(input) - 2);
    trimmed.erase(std::remove(std::begin(trimmed), std::end(trimmed), ' '), std::end(trimmed));
    trimmed.erase(std::remove(std::begin(trimmed), std::end(trimmed), '\t'), std::end(trimmed));

    std::vector<int32_t> result;
    std::istringstream ss(trimmed);
    std::string token;

    while (std::getline(ss, token, ',')) {
        if (std::empty(token)) {
            continue;
        }
        result.push_back(std::stoi(token));
    }

    return result;
}

std::vector<Interval> compute_partitions(const int32_t num_items, const int32_t num_partitions) {
    std::vector<Interval> chunks;
    const int32_t chunk_size = num_items / num_partitions;
    std::vector<int32_t> chunk_sizes(num_partitions, chunk_size);
    for (int32_t i = 0; i < (num_items % num_partitions); ++i) {
        ++chunk_sizes[i];
    }
    int32_t sum = 0;
    for (const int32_t v : chunk_sizes) {
        if (v == 0) {
            continue;
        }
        chunks.emplace_back(Interval{sum, sum + v});
        sum += v;
    }
    if (sum != num_items) {
        throw std::runtime_error{
                "Wrong sum of items divided into chunks! num_items = " + std::to_string(num_items) +
                ", num_partitions = " + std::to_string(num_partitions) +
                ", sum = " + std::to_string(sum)};
    }
    return chunks;
}

void save_tensor(const at::Tensor& tensor, const std::string& file_path) {
    const std::vector<char> pickled = torch::jit::pickle_save(tensor);
    std::ofstream fout(file_path, std::ios::out | std::ios::binary);
    fout.write(std::data(pickled), std::size(pickled));
}

}  // namespace dorado::polisher
