#include "nn/FactorisedLinearLayer.h"

#include "torch_utils/gpu_profiling.h"

#if DORADO_CUDA_BUILD
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
}

at::Tensor FactorisedLinearLayerImpl::forward(const at::Tensor x) {
    utils::ScopedProfileRange spr("factorised_linear", 2);
    return activation(up(dn(x))) * SCALE;
}

#if DORADO_CUDA_BUILD

void FactorisedLinearLayerImpl::reserve_working_memory(WorkingMemory &wm,
                                                       const AuxiliaryData *aux) {
    if (wm.layout == TensorLayout::CUTLASS_TNC_I8) {
        if (aux) {
            wm.temp({(wm.T * (int64_t)wm.N * K_) + ((int64_t)aux->NT_out_max() * C_out_)},
                    torch::kF16);
        } else {
            wm.temp({(wm.T * (int64_t)wm.N * K_) + (wm.T * (int64_t)wm.N * C_out_)}, torch::kF16);
        }
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

    auto in_bfr = wm.current.narrow(0, aux ? 3 : 2, wm.T);

    const int64_t dn_bfr_size = wm.T * (int64_t)wm.N * K_;
    const int64_t out_bfr_size =
            aux ? ((int64_t)aux->NT_out() * C_out_) : (wm.T * (int64_t)wm.N * C_out_);

    auto temp = wm.temp({dn_bfr_size + out_bfr_size}, torch::kF16);
    auto dn_bfr = temp.narrow(0, 0, dn_bfr_size);
    dn_bfr.zero_();
    auto out_bfr = temp.narrow(0, dn_bfr_size, out_bfr_size);

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

    host_factorised_linear(stream, wm.N, wm.T, parity, in_bfr.data_ptr(),
                           device_dn_weight_.data_ptr(), device_dn_weight_scale_.data_ptr(),
                           dn_bfr.data_ptr(), device_up_weight_.data_ptr(),
                           nullptr,  // bias
                           SCALE, out_layout, out_bfr.data_ptr());

    // manually update working memory
    wm.layout = TensorLayout::NTC;
    wm.C = C_out_;
    if (aux) {
        wm.N = 1;
        wm.T = aux->NT_out();
    }
    wm.current = out_bfr.view({wm.N, wm.T, wm.C});
}

#endif

}  // namespace dorado::nn
