#include "nn/LSTMStack.h"

#include "torch_utils/gpu_profiling.h"
#include "torch_utils/tensor_utils.h"

#include <stdexcept>
#include <string>

#if DORADO_CUDA_BUILD
extern "C" {
#include "koi.h"
}
#endif

#include <torch/nn.h>

namespace dorado::nn {

LSTMStackImpl::LSTMStackImpl(int num_layers, int size, bool reverse_first_)
        : layer_size(size), reverse_first(reverse_first_) {
    // torch::nn::LSTM expects/produces [N, T, C] with batch_first == true
    const auto lstm_opts = torch::nn::LSTMOptions(size, size).batch_first(true);
    for (int i = 0; i < num_layers; ++i) {
        auto label = std::string("rnn") + std::to_string(i + 1);
        rnns.emplace_back(register_module(label, torch::nn::LSTM(lstm_opts)));
    }
};

at::Tensor LSTMStackImpl::forward(at::Tensor x) {
    // Input is [N, T, C], contiguity optional
    bool is_reverse = !reverse_first;
    for (size_t i = 0; i < rnns.size(); ++i) {
        if (i != 0 || reverse_first) {
            x = x.flip(1);
            is_reverse = !is_reverse;
        }
        x = std::get<0>(rnns[i](x));
    }
    // Output is [N, T, C], contiguous
    return is_reverse ? x.flip(1) : x;
}

#if DORADO_CUDA_BUILD
void LSTMStackImpl::reserve_working_memory(WorkingMemory &wm) {
    if (wm.layout == TensorLayout::NTC) {
        wm.temp({wm.N * wm.T, 4 * layer_size}, torch::kF16);
    } else if (wm.layout == TensorLayout::CUTLASS_TNC_F16) {
        wm.next_TC(wm.T, wm.C, TensorLayout::CUTLASS_TNC_I8);
    } else if (wm.layout == TensorLayout::CUBLAS_TN2C) {
        wm.temp({wm.N, 4 * layer_size}, torch::kF16);
    }
}

void LSTMStackImpl::run_koi(WorkingMemory &wm, const AuxiliaryData *const aux) {
    utils::ScopedProfileRange spr("lstm_stack", 2);

    if (wm.layout == TensorLayout::NTC) {
        if (aux) {
            throw std::runtime_error(
                    "LSTM layer error: unsupported variable chunk sizes code path!");
        }
        return forward_quantized(wm);
    } else if (wm.layout == TensorLayout::CUBLAS_TN2C) {
        if (aux) {
            throw std::runtime_error(
                    "LSTM layer error: unsupported variable chunk sizes code path!");
        }
        return forward_cublas(wm);
    } else if (wm.layout == TensorLayout::CUTLASS_TNC_F16 ||
               wm.layout == TensorLayout::CUTLASS_TNC_I8) {
        return forward_cutlass(wm, aux);
    } else {
        throw std::runtime_error("Unhandled TensorLayout in LSTMStack.");
    }
}

void LSTMStackImpl::forward_cublas(WorkingMemory &wm) {
    // Working memory is laid out as [T+1][N][2][C] in memory, where the 2 serves to
    // interleave input and output for each LSTM layer in a specific way. The reverse LSTM
    // layers (even index) use right as input and left as output, whereas the forward
    // LSTM layers (odd index) use left as input and right as output.
    //
    // The interleaving means that x(t) and h(t-1), i.e. the input for the current timestep
    // and the output of the previous timestep, appear concatenated in memory and we can
    // perform a single matmul with the concatenated WU matrix
    // Note that both in[chunk_size][:][0][:] and in[0][:][1][:] remain
    // all zeroes, representing the initial LSTM state h(-1) in either direction.
    auto in = wm.current;
    in.index({0, torch::indexing::Slice(), 1}) = 0;
    in.index({-1, torch::indexing::Slice(), 0}) = 0;
    auto inout_all = in.flatten(2, 3);
    auto inout_left = in.narrow(0, 0, wm.T).select(2, 0);
    auto inout_right = in.narrow(0, 1, wm.T).select(2, 1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto gate_buf = wm.temp({wm.N, layer_size * 4}, torch::kF16);

    for (size_t layer_idx = 0; layer_idx < rnns.size(); ++layer_idx) {
        bool reverse = reverse_first ? !(layer_idx & 1) : (layer_idx & 1);
        utils::ScopedProfileRange spr_lstm("lstm_layer", 3);
        auto state_buf = torch::zeros({wm.N, layer_size}, in.options());
        {
            // Move weights to GPU if called for the first time
            if (device_weights.size() == layer_idx) {
                const auto &params = rnns[layer_idx]->named_parameters();
                auto w_ih = params["weight_ih_l0"];
                auto w_hh = params["weight_hh_l0"];
                device_bias.push_back(params["bias_ih_l0"].to(in.options()));
                auto weights = torch::cat({reverse ? w_hh : w_ih, reverse ? w_ih : w_hh}, 1ll);
                device_weights.push_back(weights.t().contiguous().to(in.options()));
            }

            for (int ts = 0; ts < wm.T; ++ts) {
                auto timestep_in = inout_all[reverse ? (wm.T - ts) : ts];
                auto timestep_out = reverse ? inout_left[wm.T - ts - 1] : inout_right[ts];
                // Timestep matrix multiplication
                dorado::utils::matmul_f16(timestep_in, device_weights[layer_idx], gate_buf);
                host_lstm_step_f16(stream, wm.N, layer_size, 2 * layer_size, false,
                                   device_bias[layer_idx].data_ptr(), gate_buf.data_ptr(),
                                   state_buf.data_ptr(), timestep_out.data_ptr());
            }
        }
        wm.is_input_to_rev_lstm = !reverse;
    }
}

void LSTMStackImpl::forward_cutlass(WorkingMemory &wm, const AuxiliaryData *const aux) {
    // Working memory is laid out as [T+3][N][C] in memory, where the reverse LSTM
    // layers (even index) use [1:-2] as input and [2:-1] as output, whereas the
    // forward LSTM layers (odd index) use [2:-1] as input and [1:-2] as output.
    // Note that both inout[0] and inout[-1] remain all zeroes, representing the initial
    // LSTM state h(-1) in either direction.
    if (aux && !reverse_first) {
        throw std::runtime_error(
                "LSTM layer error: unsupported first forward layer with variable chunks.");
    }

    wm.current[0] = 0;
    wm.current[wm.T + 2] = 0;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto opts_f16 = wm.current.options().dtype(torch::kF16);
    auto opts_i32 = opts_f16.dtype(torch::kI32);

    for (size_t layer_idx = 0; layer_idx < rnns.size(); ++layer_idx) {
        utils::ScopedProfileRange spr_lstm("lstm_layer", 3);
        bool reverse = reverse_first ? !(layer_idx & 1) : (layer_idx & 1);
        auto in = wm.current;
        auto type_id = (wm.layout == TensorLayout::CUTLASS_TNC_F16) ? KOI_F16 : KOI_I8;
        auto state_buf = torch::zeros({(aux ? 3 : 1) * wm.N, layer_size}, opts_f16);
        auto workspace_buf = torch::empty({1024}, opts_i32);
        constexpr int interleave = 0;

        // Move weights to GPU if called for the first time
        if (device_weights.size() == layer_idx) {
            const auto &params = rnns[layer_idx]->named_parameters();
            // Both weight tensors are tensors of size  [4 * out_size, in_size],
            // where dimension 0 is Wi|Wf|Wg|Wo stacked, so it could more accurately be
            // described as [4, outsize, in_size]. Bias is alike, with the last dim dropped.
            auto w_ih = params["weight_ih_l0"].to(opts_f16);
            auto w_hh = params["weight_hh_l0"].to(opts_f16);
            auto weights_cpu = torch::cat({reverse ? w_hh : w_ih, reverse ? w_ih : w_hh}, 1);
            auto layer_device_bias = params["bias_ih_l0"].to(opts_f16).view({4, layer_size}).t();

            if (type_id == KOI_I8) {
                auto scaled_tensor = dorado::utils::quantize_tensor(weights_cpu, 1);
                weights_cpu = scaled_tensor.t;
                auto scale = scaled_tensor.scale.view({4, layer_size}).t();
                device_scale.push_back(scale.to(opts_f16).contiguous());
            } else {
                device_scale.push_back(torch::ones_like(layer_device_bias));
            }
            device_bias.push_back(layer_device_bias.contiguous());
            // Cutlass kernel expects weights reordered as <igigigigfofofofo>
            weights_cpu = weights_cpu.view({2, 2, -1, 4, 2 * layer_size});
            auto weights_cpu_cutlass =
                    weights_cpu.permute({2, 0, 3, 1, 4}).contiguous().view({-1, 2 * layer_size});
            if (interleave) {
                weights_cpu_cutlass = weights_cpu_cutlass.view({4 * layer_size, -1, interleave})
                                              .permute({1, 0, 2});
            }
            device_weights.push_back(weights_cpu_cutlass.contiguous().to(in.device()));
        }
        void *const encoding = aux ? (reverse ? aux->device_bwd_encoding.data_ptr()
                                              : aux->device_fwd_encoding.data_ptr())
                                   : nullptr;

#if DORADO_ORIN
        int flags = utils::get_dev_opt<int>("koi_lstm_flags", 2);
#else
        int flags = utils::get_dev_opt<int>("koi_lstm_flags", 0);
#endif
        host_cutlass_lstm(stream, type_id, int(layer_idx), wm.N, layer_size,
                          wm.T + (aux && reverse), reverse ? -1 : 1, int(in.stride(1)),
                          in.data_ptr(), device_weights[layer_idx].data_ptr(),
                          device_bias[layer_idx].data_ptr(), device_scale[layer_idx].data_ptr(),
                          state_buf.data_ptr(), workspace_buf.data_ptr(), encoding, interleave,
                          flags);

        if (type_id == KOI_F16) {
            utils::ScopedProfileRange spr_convert("f16_to_int8", 4);
            auto out = wm.next_TC(wm.T, wm.C, TensorLayout::CUTLASS_TNC_I8);
            host_convert(stream, in.data_ptr(), int(in.stride(0)), int(in.stride(1)),
                         int(in.stride(2)), KOI_F16, out.data_ptr(), int(out.stride(0)),
                         int(out.stride(1)), int(out.stride(2)), KOI_I8, int(in.size(0)),
                         int(in.size(1)), int(in.size(2)));
        }

        wm.is_input_to_rev_lstm = !reverse;
    }
}

void LSTMStackImpl::forward_quantized(WorkingMemory &wm) {
    // Input and output in the same buffer, TensorLayout::NTC
    auto inout = wm.current;

    // Quantise weights and move to GPU, if called for the first time
    if (device_w_hh.empty()) {
        for (auto &rnn : rnns) {
            const auto &params = rnn->named_parameters();
            auto scaled_tensor = dorado::utils::quantize_tensor(params["weight_hh_l0"], 1);
            device_w_ih.push_back(params["weight_ih_l0"].transpose(0, 1).contiguous());
            device_w_hh.push_back(scaled_tensor.t.t().contiguous());
            device_bias.push_back(params["bias_ih_l0"]);
            device_scale.push_back(scaled_tensor.scale.contiguous());
        }
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto mm_out = wm.temp({wm.N * wm.T, 4 * layer_size}, torch::kF16);
    for (size_t i = 0; i < rnns.size(); ++i) {
        int dir = (i & 1) ? 1 : -1;
        dorado::utils::matmul_f16(inout.view({-1, layer_size}), device_w_ih[i], mm_out);
        dorado::utils::handle_cuda_result(host_small_lstm(
                stream, wm.N, wm.T, layer_size, dir, mm_out.data_ptr(), device_w_hh[i].data_ptr(),
                device_bias[i].data_ptr(), device_scale[i].data_ptr(), inout.data_ptr()));
    }
}
#endif  // if DORADO_CUDA_BUILD

}  // namespace dorado::nn