#include "tensor_utils.h"

#include "torch/torch.h"

#include <torch/csrc/jit/serialization/pickle.h>

#include <fstream>
#include <vector>

namespace dorado::utils {

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
    assert(q.dtype() == torch::kF32);

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

torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q) {
    assert(q.dtype() == torch::kF32);

    auto p = t.data_ptr<int16_t>();
    auto range_min = t.min().item<int16_t>();
    auto range_max = t.max().item<int16_t>();

    int size = t.size(0);

    std::vector<int> counts(range_max - range_min + 1, 0);
    for (int i = 0; i < size; ++i) {
        counts[p[i] - range_min]++;
    }
    std::partial_sum(counts.begin(), counts.end(), counts.begin());

    auto res = torch::empty_like(q);

    for (size_t idx = 0; idx < q.numel(); idx++) {
        int threshold = q[idx].item<float>() * (size - 1);
        for (int i = 0; i <= counts.size(); ++i) {
            if (counts[i] > threshold) {
                res[idx] = i + range_min;
                break;
            }
        }
    }

    return res;
}

}  // namespace dorado::utils
