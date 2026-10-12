#pragma once

#include "LinearLayer.h"

#include <torch/nn.h>

#include <vector>

namespace dorado::nn {

struct FactorisedLinearLayerImpl : LinearLayerImpl {
    FactorisedLinearLayerImpl(int C_in, int K, int C_out);

    at::Tensor forward(at::Tensor x) override;

#if DORADO_CUDA_BUILD
    virtual void reserve_working_memory(WorkingMemory &wm,
                                        const AuxiliaryData *aux /* = nullptr */) override;
    virtual void run_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */) override;

private:
    void forward_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);

    at::Tensor device_dn_weight_;
    at::Tensor device_dn_weight_scale_;
    at::Tensor device_up_weight_;
#endif

public:
    int C_in_;
    int K_;
    int C_out_;
    bool i8_output;

    torch::nn::Linear dn{nullptr};
    torch::nn::Linear up{nullptr};
    torch::nn::Tanh activation{nullptr};
    static constexpr float SCALE{5.f};
};

TORCH_MODULE(FactorisedLinearLayer);

}  // namespace dorado::nn
