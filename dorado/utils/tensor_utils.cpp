#include "torch/torch.h"

#include <torch/csrc/jit/serialization/pickle.h>

#include <filesystem>
#include <fstream>

namespace utils {

void serialise_tensor(torch::Tensor t, const std::string& path) {
    auto bytes = torch::jit::pickle_save(t);
    std::ofstream fout(path);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

std::vector<torch::Tensor> load_weights(const std::string& dir,
                                        const std::vector<std::string>& tensors) {
    auto weights = std::vector<torch::Tensor>();
    for (auto weight : tensors) {
        auto path = std::filesystem::path(dir) / weight;
        torch::load(weights, path.string());
    }

    return weights;
}

}  // namespace utils