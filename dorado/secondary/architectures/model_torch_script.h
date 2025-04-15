#pragma once

#include "model_torch_base.h"

#include <ATen/ATen.h>

#include <filesystem>

namespace dorado::secondary {

class ModelTorchScript : public ModelTorchBase {
public:
    ModelTorchScript(const std::filesystem::path& model_path);

    torch::Device get_device() const override;

    at::Tensor forward(at::Tensor x) override;

    void to_half() override;

    void set_normalise(const bool val) override;

    void set_eval() override;

    void to_device(torch::Device device) override;

private:
    torch::jit::script::Module m_module;
};

}  // namespace dorado::secondary
