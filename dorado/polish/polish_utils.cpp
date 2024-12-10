#include "polish_utils.h"

#include <htslib/faidx.h>
#include <spdlog/spdlog.h>

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

std::vector<int32_t> parse_int32_vector(const std::string& input) {
    if (std::empty(input)) {
        return {};
    }
    if ((std::size(input) < 2) || (input.front() != '[') || (input.back() != ']')) {
        throw std::runtime_error("Input string must start with '[' and end with ']'.");
    }

    // Remove the brackets and trim the string
    std::string trimmed = input.substr(1, input.size() - 2);
    trimmed.erase(std::remove(trimmed.begin(), trimmed.end(), ' '), trimmed.end());
    trimmed.erase(std::remove(trimmed.begin(), trimmed.end(), '\t'), trimmed.end());

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

std::string fetch_seq(const std::filesystem::path& index_fn,
                      const std::string& seq_name,
                      int32_t start,
                      int32_t end) {
    std::unique_ptr<faidx_t, decltype(&fai_destroy)> fai(fai_load(index_fn.string().c_str()),
                                                         fai_destroy);

    if (!fai) {
        spdlog::error("Failed to load index for file: '{}'.", index_fn.string());
        return {};
    }

    const int32_t seq_len = faidx_seq_len(fai.get(), seq_name.c_str());

    start = std::max(start, 0);
    end = (end < 0) ? seq_len : std::min(end, seq_len);

    if (end <= start) {
        spdlog::error(
                "Cannot load sequence because end <= start! seq_name = {}, start = {}, end = {}.",
                seq_name, start, end);
        return {};
    }

    // Get the sequence.
    int32_t temp_seq_len = 0;
    std::unique_ptr<char, decltype(&free)> seq(
            faidx_fetch_seq(fai.get(), seq_name.c_str(), start, end - 1, &temp_seq_len), free);

    if (temp_seq_len != (end - start)) {
        spdlog::error(
                "Loaded sequence length does not match the specified interval! seq_name = {}, "
                "start = {}, end = {}, loaded len = {}.",
                seq_name, start, end, temp_seq_len);
        return {};
    }

    std::string ret;
    if (seq) {
        ret = std::string(seq.get(), temp_seq_len);
    }

    return ret;
}

void save_tensor(const torch::Tensor& tensor, const std::string& file_path) {
    auto pickled = torch::pickle_save(tensor);
    std::ofstream fout(file_path, std::ios::out | std::ios::binary);
    fout.write(std::data(pickled), std::size(pickled));
}

}  // namespace dorado::polisher
