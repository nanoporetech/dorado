#include "tensor_utils.h"

#include "torch/torch.h"

#include <torch/csrc/jit/serialization/pickle.h>

#include <fstream>

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
    assert(t.dtype().name() == "float");
    assert(q.dtype().name() == "float");
    if (!torch::equal(q, std::get<0>(q.sort()))) {
        throw std::runtime_error("quantiles q are not sorted");
    }

    auto tmp = t.clone();
    auto res = torch::empty_like(q);

    auto start = tmp.data_ptr<float>();
    auto end = tmp.data_ptr<float>() + tmp.size(0);

    for (int i = 0; i < q.size(0); i++) {
        auto m =
                tmp.data_ptr<float>() + static_cast<size_t>((tmp.size(0) - 1) * q[i].item<float>());
        std::nth_element(start, m, end);
        res[i] = *m;
        start = m;
    }

    return res;
}

}  // namespace utils
