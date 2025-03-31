#pragma once

#include "config/BasecallModelConfig.h"

#include <torch/nn.h>

#include <vector>

namespace dorado::basecall::nn {

#if DORADO_CUDA_BUILD
class WorkingMemory;
enum class TensorLayout { NTC, TNC, CUTLASS_TNC_F16, CUTLASS_TNC_I8, CUBLAS_TN2C };
#endif

struct ConvStackImpl : torch::nn::Module {
    explicit ConvStackImpl(const std::vector<config::ConvParams> &layer_params);
#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm);
    void run_koi(WorkingMemory &wm);
#endif  // if DORADO_CUDA_BUILD

    at::Tensor forward(at::Tensor x);

    struct ConvLayer {
        explicit ConvLayer(const config::ConvParams &params);
        const config::ConvParams params;
        torch::nn::Conv1d conv{nullptr};
#if DORADO_CUDA_BUILD
        TensorLayout output_layout{TensorLayout::NTC};
        bool cutlass_conv{false};
        int output_T_padding{0};
        at::Tensor w_device;
        at::Tensor b_device;

        void reserve_working_memory(WorkingMemory &wm);
        void run_koi(WorkingMemory &wm);
#endif  // if DORADO_CUDA_BUILD
    };

    std::vector<ConvLayer> layers;
};

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
TORCH_MODULE(ConvStack);
TORCH_MODULE(Clamp);

struct CRFModelImpl : torch::nn::Module {
    explicit CRFModelImpl(const config::BasecallModelConfig &config);
    void load_state_dict(const std::vector<at::Tensor> &weights);
#if DORADO_CUDA_BUILD
    at::Tensor run_koi(const at::Tensor &in);
#endif

    at::Tensor forward(const at::Tensor &x);
    ConvStack convs{nullptr};
    LSTMStack rnns{nullptr};
    LinearCRF linear1{nullptr}, linear2{nullptr};
    Clamp clamp1{nullptr};
    torch::nn::Sequential encoder{nullptr};
};

TORCH_MODULE(CRFModel);

}  // namespace dorado::basecall::nn
