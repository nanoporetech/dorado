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

std::vector<torch::Tensor> load_tensors(const std::string& dir,
                                        const std::vector<std::string>& tensors) {
    auto weights = std::vector<torch::Tensor>();
    for (auto tensor : tensors) {
        auto path = std::filesystem::path(dir) / tensor;
        torch::load(weights, path.string());
    }

    return weights;
}

}  // namespace utils