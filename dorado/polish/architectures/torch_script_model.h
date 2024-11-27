#pragma once

#include "torch_model_base.h"

#include <torch/torch.h>

#include <filesystem>

namespace dorado::polisher {

class TorchScriptModel : public TorchModel {
public:
    TorchScriptModel(const std::filesystem::path& model_path);

    torch::Tensor forward(torch::Tensor x) override;

    void to_half() override;

    void set_eval() override;

    void to_device(torch::Device device, bool non_blocking = false) override;

private:
    torch::jit::script::Module m_module;
};

}  // namespace dorado::polisher
