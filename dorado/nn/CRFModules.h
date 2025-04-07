#pragma once

#include "WorkingMemory.h"

#include <torch/nn.h>

namespace dorado::nn {

struct LinearCRFImpl : torch::nn::Module {
    LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale);
    at::Tensor forward(const at::Tensor &x);
#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm);
    void run_koi(WorkingMemory &wm);
    at::Tensor w_device;
    at::Tensor weight_scale;
#endif  // if DORADO_CUDA_BUILD
    bool bias;
    static constexpr int scale = 5;
    torch::nn::Linear linear{nullptr};
    torch::nn::Tanh activation{nullptr};
};

struct LSTMStackImpl : torch::nn::Module {
    LSTMStackImpl(int num_layers, int size);
    at::Tensor forward(at::Tensor x);
#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm);
    void run_koi(WorkingMemory &wm);

private:
    void forward_cublas(WorkingMemory &wm);
    void forward_cutlass(WorkingMemory &wm);
    void forward_quantized(WorkingMemory &wm);

    std::vector<at::Tensor> device_weights;
    std::vector<at::Tensor> device_w_ih;
    std::vector<at::Tensor> device_w_hh;
    std::vector<at::Tensor> device_bias;
    std::vector<at::Tensor> device_scale;
#endif  // if DORADO_CUDA_BUILD
    int layer_size;
    std::vector<torch::nn::LSTM> rnns;
};

struct ClampImpl : torch::nn::Module {
    ClampImpl(float _min, float _max, bool _active);
    at::Tensor forward(at::Tensor x);
    bool active;
    float min, max;
};

TORCH_MODULE(LSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(Clamp);

}  // namespace dorado::nn