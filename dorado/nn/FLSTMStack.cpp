#include "nn/FLSTMStack.h"

#include "torch_utils/tensor_utils.h"

#if DORADO_CUDA_BUILD
#include "torch_utils/gpu_profiling.h"

#include <c10/cuda/CUDAStream.h>

extern "C" {
#include "koi.h"
}
#else
#include <tuple>
#endif

namespace dorado::nn {

FLSTMLayerImpl::FLSTMLayerImpl(const int C, const int K, at::TensorOptions opts) : C_(C) {
    dn_weight_ih_ = register_parameter("dn_weight_ih", torch::empty({K, C}, opts));
    dn_weight_hh_ = register_parameter("dn_weight_hh", torch::empty({K, C}, opts));
    up_weight_ih_ = register_parameter("up_weight_ih", torch::empty({4 * C, K}, opts));
    up_weight_hh_ = register_parameter("up_weight_hh", torch::empty({4 * C, K}, opts));
    up_bias_ih_ = register_parameter("up_bias_ih", torch::empty({4 * C}, opts));
    up_bias_hh_ = register_parameter("up_bias_hh", torch::empty({4 * C}, opts));
}

at::Tensor FLSTMLayerImpl::forward(at::Tensor x) {
    const auto opts = x.options();

    x = x.transpose(0, 1);  // NTC -> TNC
    const int T = x.size(0);
    const int N = x.size(1);

    at::Tensor hh = torch::empty({T + 1, N, C_}, opts);
    hh[0] = 0;

    at::Tensor c = torch::zeros({N, C_}, opts);

    const auto sigmoid_hard = [](const at::Tensor &a) {
        return a.mul_(0.2f).add_(0.5f).clamp_(0.f, 1.f);
    };
    const auto tanh_hard = [](const at::Tensor &a) { return a.clamp_(-1.f, 1.f); };

    for (int t = 0; t < T; ++t) {
        auto ih_t = torch::matmul(torch::matmul(x[t], dn_weight_ih_.t()), up_weight_ih_.t())
                            .add_(up_bias_ih_);
        auto gates = torch::matmul(torch::matmul(hh[t], dn_weight_hh_.t()), up_weight_hh_.t())
                             .add_(up_bias_hh_)
                             .add_(ih_t)
                             .chunk(4, 1);
        auto i = sigmoid_hard(gates[0]);
        auto f = sigmoid_hard(gates[1]);
        auto g = tanh_hard(gates[2]);
        auto o = sigmoid_hard(gates[3]);
        c = (f * c) + (i * g);
        hh[t + 1] = o * torch::tanh(c);
    }

    using namespace torch::indexing;
    hh = hh.index({Slice(1, None), Slice(), Slice()}).transpose(0, 1);

    return hh;
}

FLSTMStackImpl::FLSTMStackImpl(const int num_layers,
                               const int C,
                               const int K,
                               const bool first_reverse,
                               at::TensorOptions opts)
        : C_(C), K_(K), first_reverse_(first_reverse) {
#if !DORADO_CUDA_BUILD
    // These are only used in the CUDA path.
    std::ignore = std::make_tuple(C_, K_, first_reverse_);
#endif
    for (int i = 0; i < num_layers; ++i) {
        const auto label = std::string{"rnn"} + std::to_string(i + 1);
        layers_.emplace_back(register_module(label, FLSTMLayer(C, K, opts)));
    }
}

at::Tensor FLSTMStackImpl::forward(at::Tensor x) {
    bool is_reverse = !first_reverse_;
    for (int i = 0; i < std::ssize(layers_); ++i) {
        if ((i > 0) || first_reverse_) {
            utils::flip_inplace(x, 1);
            is_reverse ^= 1;
        }
        x = layers_[i]->forward(x);
    }
    if (is_reverse) {
        utils::flip_inplace(x, 1);
    }
    return x.contiguous();
}

#if DORADO_CUDA_BUILD

void FLSTMStackImpl::reserve_working_memory(WorkingMemory &wm) {
    if (wm.layout == TensorLayout::CUTLASS_TNC_I8) {
        wm.temp({(2 * (wm.T + 1) * (int64_t)wm.N * (2 * K_)) +  // char
                 (4 * ((wm.T + 1) * (int64_t)wm.N)) +           // float
                 (4 * ((int64_t)wm.N / 2U))},                   // int
                torch::kU8);
    } else if ((wm.layout == TensorLayout::CUBLAS_TNC) ||
               (wm.layout == TensorLayout::CUTLASS_TNC_F16)) {
        wm.temp({wm.N * ((2 * K_) + (4 * C_))}, torch::kF16);
    } else {
        throw std::runtime_error("FLSTMStack error: unsupported TensorLayout!");
    }
}

void FLSTMStackImpl::run_koi(WorkingMemory &wm, const AuxiliaryData *aux) {
    if (wm.layout == TensorLayout::CUTLASS_TNC_I8) {
        forward_koi(wm, aux);
    } else if ((wm.layout == TensorLayout::CUBLAS_TNC) ||
               (wm.layout == TensorLayout::CUTLASS_TNC_F16)) {
        if (aux) {
            throw std::runtime_error("FLSTMStack error: unsupported variable chunks path!");
        }
        forward_cublas(wm);
    } else {
        throw std::runtime_error("FLSTMStack error: unsupported TensorLayout!");
    }
}

void FLSTMStackImpl::forward_koi(WorkingMemory &wm, const AuxiliaryData *aux) {
    if (aux && !first_reverse_) {
        throw std::runtime_error(
                "FLSTM stack error: unsupported first forward layer with variable chunks.");
    }
    if (aux && ((std::size(layers_) % 2U) == 0)) {
        throw std::runtime_error(
                "FLSTM stack error: unsupported even number of layers with variable chunks.");
    }

    auto inout = wm.current;

    if (aux == nullptr) {
        inout[0] = 0;
        inout[wm.T + 1] = 0;
        inout[wm.T + 2] = 1;
    } else {
        inout[wm.T + 2] = 0;
        inout[wm.T + 3] = 0;
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_f16 = wm.current.options().dtype(torch::kF16);

    const int64_t dn_bfr_size = 2 * (wm.T + 1) * (int64_t)wm.N * (2 * K_);
    const int64_t dn_scale_bfr_size = 4 * ((wm.T + 1) * (int64_t)wm.N);
    const int64_t workspace_bfr_size = 4 * ((int64_t)wm.N / 2U);

    auto temp_bfr = wm.temp({dn_bfr_size + dn_scale_bfr_size + workspace_bfr_size}, torch::kU8);
    temp_bfr.zero_();

    auto dn_bfr = temp_bfr.narrow(0, 0, dn_bfr_size);
    auto dn_scale_bfr = temp_bfr.narrow(0, dn_bfr_size, dn_scale_bfr_size);
    auto workspace_bfr = temp_bfr.narrow(0, dn_bfr_size + dn_scale_bfr_size, workspace_bfr_size);

    for (int layer = 0; layer < std::ssize(layers_); ++layer) {
        utils::ScopedProfileRange spr_lstm("flstm_layer", 3);

        const bool reverse = first_reverse_ ^ (layer & 1);

        if (std::ssize(device_up_weights_) == layer) {  // move weights to GPU first time around
            const auto &params = layers_[layer]->named_parameters();

            auto scaled_dn_weights_ih =
                    utils::quantize_tensor(params["dn_weight_ih"].to(opts_f16), 1);

            scaled_dn_weights_ih.t = scaled_dn_weights_ih.t.view({-1, 2, 2, 2, 2, 2, C_})
                                             .permute({2, 0, 4, 1, 3, 5, 6})
                                             .contiguous()
                                             .view({K_, C_});
            scaled_dn_weights_ih.scale = scaled_dn_weights_ih.scale.view({-1, 2, 2, 2, 2, 2})
                                                 .permute({2, 0, 4, 1, 3, 5})
                                                 .contiguous()
                                                 .view({K_});

            device_dn_weights_ih_.push_back(scaled_dn_weights_ih.t);
            device_dn_weights_scale_ih_.push_back(scaled_dn_weights_ih.scale.to(opts_f16));

            auto scaled_dn_weights_hh =
                    utils::quantize_tensor(params["dn_weight_hh"].to(opts_f16), 1);

            scaled_dn_weights_hh.t = scaled_dn_weights_hh.t.view({-1, 2, 2, 2, 2, 2, C_})
                                             .permute({2, 0, 4, 1, 3, 5, 6})
                                             .contiguous()
                                             .view({K_, C_});
            scaled_dn_weights_hh.scale = scaled_dn_weights_hh.scale.view({-1, 2, 2, 2, 2, 2})
                                                 .permute({2, 0, 4, 1, 3, 5})
                                                 .contiguous()
                                                 .view({K_});

            device_dn_weights_hh_.push_back(scaled_dn_weights_hh.t);
            device_dn_weights_scale_hh_.push_back(scaled_dn_weights_hh.scale.to(opts_f16));

            auto up_weights_ih = params["up_weight_ih"].to(opts_f16);
            auto up_weights_hh = params["up_weight_hh"].to(opts_f16);
            auto up_weights = torch::cat({up_weights_ih, up_weights_hh}, 1);

            auto scaled_up_weights = utils::quantize_tensor(up_weights, 1);
            device_up_weights_.push_back(scaled_up_weights.t.view({4, -1, 2, 2, 2, 2, 2 * K_})
                                                 .permute({1, 5, 0, 2, 3, 4, 6})
                                                 .contiguous()
                                                 .view({-1, 2 * K_}));
            device_up_weights_scale_.push_back(scaled_up_weights.scale.to(opts_f16)
                                                       .view({4, -1, 2, 2, 2, 2})
                                                       .permute({1, 5, 2, 3, 4, 0})
                                                       .contiguous()
                                                       .view({-1}));

            auto up_bias = params["up_bias_ih"].to(opts_f16);
            device_up_bias_.push_back(up_bias.view({4, -1, 2, 2, 2, 2})
                                              .permute({1, 5, 2, 3, 4, 0})
                                              .contiguous()
                                              .view({-1}));
        }

        const int parity = (1 - (layer % 2U));

        void *const encoding = aux ? (reverse ? aux->device_bwd_encoding.data_ptr()
                                              : aux->device_fwd_encoding.data_ptr())
                                   : nullptr;

        host_factorised_lstm(
                stream, wm.N, wm.T + (aux && reverse), reverse ? -1 : 1, parity, inout.data_ptr(),
                workspace_bfr.data_ptr(), encoding, device_dn_weights_ih_[layer].data_ptr(),
                device_dn_weights_scale_ih_[layer].data_ptr(),
                device_dn_weights_hh_[layer].data_ptr(),
                device_dn_weights_scale_hh_[layer].data_ptr(), dn_bfr.data_ptr(),
                dn_scale_bfr.data_ptr(), device_up_weights_[layer].data_ptr(),
                device_up_weights_scale_[layer].data_ptr(), device_up_bias_[layer].data_ptr());

        wm.is_input_to_rev_lstm = !reverse;
    }
}

void FLSTMStackImpl::forward_cublas(WorkingMemory &wm) {
    auto inout = wm.current;
    inout[0] = 0;
    inout[wm.T + 2] = 0;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const int64_t dn_bfr_size = wm.N * (2 * K_);
    const int64_t up_bfr_size = wm.N * (4 * C_);

    auto temp_bfr = wm.temp({dn_bfr_size + up_bfr_size}, torch::kF16);

    auto dn_bfr = temp_bfr.narrow(0, 0, dn_bfr_size).view({wm.N, 2, K_});
    auto dn_ih_bfr = dn_bfr.select(1, 0);
    auto dn_hh_bfr = dn_bfr.select(1, 1);
    dn_bfr = dn_bfr.view({wm.N, -1});

    auto up_bfr = temp_bfr.narrow(0, dn_bfr_size, up_bfr_size).view({wm.N, 4 * C_});

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
            host_lstm_step_f16(stream, wm.N, C_, C_, true, device_up_bias_[layer].data_ptr(),
                               up_bfr.data_ptr(), state_bfr.data_ptr(), inout[t_o].data_ptr());
        }

        wm.is_input_to_rev_lstm = !reverse;
    }
}

#endif

}  // namespace dorado::nn