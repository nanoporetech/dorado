#pragma once

#include "AuxiliaryData.h"
#include "WorkingMemory.h"

#include <torch/nn.h>

#include <memory>

namespace dorado::nn {

struct LinearCRFImpl : torch::nn::Module {
    LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale);
    at::Tensor forward(const at::Tensor &x);
#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
    void run_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
    at::Tensor w_device;
    at::Tensor weight_scale;
#endif  // if DORADO_CUDA_BUILD
    bool bias;
    static constexpr int scale = 5;
    torch::nn::Linear linear{nullptr};
    torch::nn::Tanh activation{nullptr};
};

struct ClampImpl : torch::nn::Module {
    ClampImpl(float _min, float _max, bool _active);
    at::Tensor forward(at::Tensor x);
    bool active;
    float min, max;
};

TORCH_MODULE(LinearCRF);
TORCH_MODULE(Clamp);

}  // namespace dorado::nn