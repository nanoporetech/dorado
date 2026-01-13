#include "nn/FLSTMStack.h"

#include "torch_utils/gpu_profiling.h"

#include <stdexcept>
#include <tuple>

#if DORADO_CUDA_BUILD

extern "C" {
#include "koi.h"
}

#endif

namespace dorado::nn {

FLSTMLayerImpl::FLSTMLayerImpl(const int C, const int K) {
    dn_weight_ih_ = register_parameter("dn_weight_ih", torch::empty({K, C}));
    dn_weight_hh_ = register_parameter("dn_weight_hh", torch::empty({K, C}));
    up_weight_ih_ = register_parameter("up_weight_ih", torch::empty({4 * C, K}));
    up_weight_hh_ = register_parameter("up_weight_hh", torch::empty({4 * C, K}));
    up_bias_ih_ = register_parameter("up_bias_ih", torch::empty({4 * C}));
    up_bias_hh_ = register_parameter("up_bias_hh", torch::empty({4 * C}));
}

at::Tensor FLSTMLayerImpl::forward(at::Tensor x) {
    throw std::runtime_error("FLSTMLayer::forward is not supported!");
    x = x * 1;  // clang-tidy
}

FLSTMStackImpl::FLSTMStackImpl(const int num_layers,
                               const int C,
                               const int K,
                               const bool first_reverse)
        : C_(C), K_(K), first_reverse_(first_reverse) {
#if !DORADO_CUDA_BUILD
    // These are only used in the CUDA path.
    std::ignore = std::make_tuple(C_, K_, first_reverse_);
#endif
    for (int i = 0; i < num_layers; ++i) {
        const auto label = std::string{"rnn"} + std::to_string(i + 1);
        layers_.emplace_back(register_module(label, FLSTMLayer(C, K)));
    }
}

at::Tensor FLSTMStackImpl::forward(at::Tensor x) {
    throw std::runtime_error("FLSTMStack::forward is not supported!");
    x = x * 1;  // clang-tidy
}

#if DORADO_CUDA_BUILD

void FLSTMStackImpl::reserve_working_memory(WorkingMemory &wm) {
    if ((wm.layout == TensorLayout::CUBLAS_TNC) || (wm.layout == TensorLayout::CUTLASS_TNC_F16)) {
        wm.temp({wm.N * ((2 * K_) + (4 * C_))}, torch::kF16);
    } else {
        throw std::runtime_error("FLSTMStack error: unsupported TensorLayout!");
    }
}

void FLSTMStackImpl::run_koi(WorkingMemory &wm, const AuxiliaryData *aux) {
    if (aux) {
        throw std::runtime_error("FLSTMStack error: variable chunks are not supported!");
    }
    if ((wm.layout == TensorLayout::CUBLAS_TNC) || (wm.layout == TensorLayout::CUTLASS_TNC_F16)) {
        return forward_cublas(wm);
    } else {
        throw std::runtime_error("FLSTMStack error: unsupported TensorLayout!");
    }
}

void FLSTMStackImpl::forward_cublas(WorkingMemory &wm) {
    auto inout = wm.current;
    inout[0] = 0;
    inout[wm.T + 2] = 0;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto temp_bfr = wm.temp({wm.N * ((2 * K_) + (4 * C_))}, torch::kF16);

    auto dn_bfr = temp_bfr.narrow(0, 0, wm.N * (2 * K_)).view({wm.N, 2, K_});
    auto dn_ih_bfr = dn_bfr.select(1, 0);
    auto dn_hh_bfr = dn_bfr.select(1, 1);
    dn_bfr = dn_bfr.view({wm.N, -1});

    auto up_bfr = temp_bfr.narrow(0, wm.N * (2 * K_), wm.N * (4 * C_)).view({wm.N, 4 * C_});

    const bool hard_activation = utils::get_dev_opt<bool>("koi_use_hard_act", false);

    for (int layer = 0; layer < std::ssize(layers_); ++layer) {
        utils::ScopedProfileRange spr_lstm("flstm_layer", 3);

        const bool reverse = first_reverse_ ^ (layer & 1);
        auto state_bfr = torch::zeros({wm.N, C_}, inout.options());

        if (std::ssize(device_up_weights_) == layer) {  // move weights to GPU first time around
            const auto &params = layers_[layer]->named_parameters();
            device_dn_weights_ih_.push_back(
                    params["dn_weight_ih"].t().contiguous().to(inout.options()));
            device_dn_weights_hh_.push_back(
                    params["dn_weight_hh"].t().contiguous().to(inout.options()));
            auto up_weight_ih = params["up_weight_ih"].to(inout.options());
            auto up_weight_hh = params["up_weight_hh"].to(inout.options());
            auto up_weight = torch::cat({up_weight_ih, up_weight_hh}, 1);
            device_up_weights_.push_back(up_weight.t().contiguous());
            device_up_bias_.push_back(params["up_bias_ih"].to(inout.options()));
        }

        for (int t = 0; t < wm.T; ++t) {
            const int t_i = reverse ? (wm.T - t) : (2 + t);
            const int t_o = t_i + (reverse ? 1 : -1);
            const int t_h = t_i + (reverse ? 2 : -2);

            // down projection
            utils::matmul_f16(inout[t_i], device_dn_weights_ih_[layer], dn_ih_bfr);
            utils::matmul_f16(inout[t_h], device_dn_weights_hh_[layer], dn_hh_bfr);

            // up projection
            utils::matmul_f16(dn_bfr, device_up_weights_[layer], up_bfr);

            // gate calculation
            host_lstm_step_f16(stream, wm.N, C_, C_, hard_activation,
                               device_up_bias_[layer].data_ptr(), up_bfr.data_ptr(),
                               state_bfr.data_ptr(), inout[t_o].data_ptr());
        }

        wm.is_input_to_rev_lstm = !reverse;  // needed?
    }
}

#endif

}  // namespace dorado::nn