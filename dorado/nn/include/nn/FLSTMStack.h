#pragma once

#include "RNNStack.h"

#include <torch/nn.h>

#include <vector>

namespace dorado::nn {

struct FLSTMLayerImpl : torch::nn::Module {
    FLSTMLayerImpl(int C, int K);

    at::Tensor forward(at::Tensor x);

private:
    at::Tensor dn_weight_ih_;
    at::Tensor dn_weight_hh_;
    at::Tensor up_weight_ih_;
    at::Tensor up_weight_hh_;
    at::Tensor up_bias_ih_;
    at::Tensor up_bias_hh_;
};

TORCH_MODULE(FLSTMLayer);

struct FLSTMStackImpl : RNNStackImpl {
    FLSTMStackImpl(int num_layers, int C, int K, bool first_reverse);

    at::Tensor forward(at::Tensor x) override;

#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm) override;
    void run_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */) override;

private:
    void forward_cublas(WorkingMemory &wm);

    std::vector<at::Tensor> device_dn_weights_ih_;
    std::vector<at::Tensor> device_dn_weights_hh_;
    std::vector<at::Tensor> device_up_weights_;
    std::vector<at::Tensor> device_up_bias_;
#endif

private:
    int C_;
    int K_;
    std::vector<FLSTMLayer> layers_;
    bool first_reverse_;
};

TORCH_MODULE(FLSTMStack);

}  // namespace dorado::nn