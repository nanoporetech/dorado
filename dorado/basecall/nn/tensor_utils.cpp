#include "tensor_utils.h"

#include "spdlog/spdlog.h"

#include <torch/serialize.h>

#include <cstdlib>
#include <filesystem>

namespace dorado::basecall::nn {
std::string shape(const at::Tensor &t, const std::string &name) {
    std::string str = name + ".shape()={";
    const auto &sz = t.sizes();
    for (size_t i = 0; i < sz.size(); ++i) {
        if (i != 0) {
            str += ", ";
        }
        str += std::to_string(sz[i]);
    }
    str += "}";
    return str;
}

void dump_tensor(const at::Tensor &t, const std::string &name) {
    const char *envVar = std::getenv("DORADO_TENSOR_DUMP");
    if (envVar != nullptr) {
        return;
    }
    const auto fp = std::filesystem::current_path() / (name + ".pt");
    spdlog::debug("Saving tensor: {}", fp.u8string());
    torch::save(t, fp);
}
}  // namespace dorado::basecall::nn