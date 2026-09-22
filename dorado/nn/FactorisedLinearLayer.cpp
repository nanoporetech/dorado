#include "nn/FactorisedLinearLayer.h"

#include "torch_utils/gpu_profiling.h"

#if DORADO_CUDA_BUILD
#include "torch_utils/tensor_utils.h"

extern "C" {
#include "koi.h"
}
#endif

namespace dorado::nn {

FactorisedLinearLayerImpl::FactorisedLinearLayerImpl(int C_in, int K, int C_out)
        : C_in_(C_in), K_(K), C_out_(C_out) {
    dn = register_module("up", torch::nn::Linear(torch::nn::LinearOptions(C_in, K).bias(false)));
    up = register_module("dn", torch::nn::Linear(torch::nn::LinearOptions(K, C_out).bias(false)));
    activation = register_module("activation", torch::nn::Tanh());
    i8_output = utils::get_dev_opt("linear_i8", true);
}

at::Tensor FactorisedLinearLayerImpl::forward(const at::Tensor x) {
    utils::ScopedProfileRange spr("factorised_linear", 2);
    return activation(up(dn(x))) * SCALE;
}

#if DORADO_CUDA_BUILD

void FactorisedLinearLayerImpl::reserve_working_memory(WorkingMemory &wm,
                                                       const AuxiliaryData *aux) {
    if (wm.layout == TensorLayout::CUTLASS_TNC_I8) {
        int N_in = wm.N;
        int T_in = wm.T;
        wm.temp({T_in, N_in, K_}, torch::kF16);
        int N_out = aux ? 1 : N_in;
        int T_out = aux ? aux->NT_out_max() : T_in;
        auto wm_out_layout = i8_output ? TensorLayout::NTC_I8 : TensorLayout::NTC;
        wm.next_NTC(N_out, T_out, C_out_, wm_out_layout);
    } else {
        throw std::runtime_error("FactorisedLinearLayer error: unsupported TensorLayout!");
    }
}

void FactorisedLinearLayerImpl::run_koi(WorkingMemory &wm, const AuxiliaryData *aux) {
    if (wm.layout == TensorLayout::CUTLASS_TNC_I8) {
        forward_koi(wm, aux);
    } else {
        throw std::runtime_error("FactorisedLinearLayer error: unsupported TensorLayout!");
    }
}

void FactorisedLinearLayerImpl::forward_koi(WorkingMemory &wm, const AuxiliaryData *aux) {
    utils::ScopedProfileRange spr("factorised_linear", 2);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_f16 = wm.current.options().dtype(torch::kF16);

    int N_in = wm.N;
    int T_in = wm.T;
    auto in_bfr = wm.current.narrow(0, aux ? 3 : 2, T_in);

    auto dn_bfr = wm.temp({T_in, N_in, K_}, torch::kF16);
    dn_bfr.zero_();
    int N_out = aux ? 1 : N_in;
    int T_out = aux ? aux->NT_out() : T_in;
    auto wm_out_layout = i8_output ? TensorLayout::NTC_I8 : TensorLayout::NTC;
    at::Tensor out_bfr = wm.next_NTC(N_out, T_out, C_out_, wm_out_layout);

    if (!device_dn_weight_.defined()) {
        auto dn_weight = utils::quantize_tensor(dn->weight.to(opts_f16), 1);

        device_dn_weight_ = dn_weight.t.view({-1, 2, 2, 2, 2, C_in_})
                                    .permute({0, 3, 1, 2, 4, 5})
                                    .contiguous()
                                    .view({K_, C_in_});
        device_dn_weight_scale_ = dn_weight.scale.to(opts_f16)
                                          .view({-1, 2, 2, 2, 2})
                                          .permute({0, 3, 1, 2, 4})
                                          .contiguous()
                                          .view({K_});

        auto up_weight = up->weight.to(opts_f16);

        device_up_weight_ = up_weight.view({-1, 2, 2, 2, 2, 2, K_})
                                    .permute({0, 3, 4, 1, 2, 5, 6})
                                    .contiguous()
                                    .view({C_out_, K_});
    }

    const int parity = 1;
    void *const out_layout = aux ? aux->device_out_layout.data_ptr() : nullptr;

    KoiTypeId out_koi_type = i8_output ? KOI_I8 : KOI_F16;
    float scale = i8_output ? 127.f : SCALE;
    host_factorised_linear(stream, out_koi_type, N_in, T_in, parity, in_bfr.data_ptr(),
                           device_dn_weight_.data_ptr(), device_dn_weight_scale_.data_ptr(),
                           dn_bfr.data_ptr(), device_up_weight_.data_ptr(),
                           nullptr,  // bias
                           scale, out_layout, out_bfr.data_ptr());
}

#endif

}  // namespace dorado::nn
