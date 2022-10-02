#include "tensor_utils.h"

#include "torch/torch.h"

#include <torch/csrc/jit/serialization/pickle.h>

#include <array>
#include <fstream>
#include <vector>

namespace utils {

void serialise_tensor(torch::Tensor t, const std::string& path) {
    auto bytes = torch::jit::pickle_save(t);
    std::ofstream fout(path);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

std::vector<torch::Tensor> load_tensors(const std::filesystem::path& dir,
                                        const std::vector<std::string>& tensors) {
    auto weights = std::vector<torch::Tensor>();
    for (auto tensor : tensors) {
        auto path = dir / tensor;
        torch::load(weights, path.string());
    }

    return weights;
}

torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q) {
    assert(q.dtype().name() == "float");

    auto tmp = t.clone();
    auto [qval, qidx] = q.sort();
    auto res = torch::empty_like(q);

    auto start = tmp.data_ptr<float>();
    auto end = tmp.data_ptr<float>() + tmp.size(0);

    for (int i = 0; i < q.size(0); i++) {
        auto m = tmp.data_ptr<float>() +
                 static_cast<size_t>((tmp.size(0) - 1) * qval[i].item<float>());
        std::nth_element(start, m, end);
        res[qidx[i]] = *m;
        start = m;
    }

    return res;
}

torch::Tensor quantile_radix(const torch::Tensor t, const torch::Tensor q) {
    assert(q.dtype().name() == "float");

    auto p = t.data_ptr<int>();
    std::vector<int> tmp{p, p + t.size(0)};

    auto res = torch::empty_like(q);

    int radix = 1;

    // Largest element in unsorted array
    int max = *(std::max_element(tmp.begin(), tmp.end()));

    while (max / radix) {
        std::array<std::vector<int>, 10> buckets;

        for (const auto& num : tmp) {
            int digit = num / radix % 10;
            buckets[digit].push_back(num);
        }

        size_t k = 0;

        // Take the elements out of buckets into the array
        for (size_t i = 0; i < 10; i++) {
            for (size_t j = 0; j < buckets[i].size(); j++) {
                tmp[k] = buckets[i][j];
                k++;
            }
        }

        // Change the place of digit used for sorting
        radix *= 10;
    }

    for (size_t idx = 0; idx < q.numel(); idx++) {
        res[idx] = tmp[static_cast<size_t>(q[idx].item<float>() * tmp.size())];
    }

    return res;
}

}  // namespace utils
