#pragma once

#include "AuxiliaryData.h"
#include "RNNStack.h"
#include "WorkingMemory.h"

#include <torch/nn.h>

#include <memory>
#include <vector>

namespace dorado::nn {

struct LSTMStackImpl : RNNStackImpl {
    LSTMStackImpl(int num_layers, int size, bool reverse_first);
    at::Tensor forward(at::Tensor x) override;
#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm) override;
    void run_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */) override;

private:
    void forward_cublas(WorkingMemory &wm);
    void forward_cutlass(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
    void forward_quantized(WorkingMemory &wm);

    std::vector<at::Tensor> device_weights;
    std::vector<at::Tensor> device_w_ih;
    std::vector<at::Tensor> device_w_hh;
    std::vector<at::Tensor> device_bias;
    std::vector<at::Tensor> device_scale;
#endif  // if DORADO_CUDA_BUILD

    int layer_size;
    std::vector<torch::nn::LSTM> rnns;
    const bool reverse_first;
};

TORCH_MODULE(LSTMStack);

}  // namespace dorado::nn