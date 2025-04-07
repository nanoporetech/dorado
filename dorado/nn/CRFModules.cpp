#include "CRFModules.h"

#include "nn/KoiUtils.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/tensor_utils.h"

#if DORADO_CUDA_BUILD
extern "C" {
#include "koi.h"
}
#endif

namespace dorado::nn {

LinearCRFImpl::LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale)
        : bias(bias_) {
    linear = register_module(
            "linear", torch::nn::Linear(torch::nn::LinearOptions(insize, outsize).bias(bias)));
    if (tanh_and_scale) {
        activation = register_module("activation", torch::nn::Tanh());
    }
};

at::Tensor LinearCRFImpl::forward(const at::Tensor &x) {
    utils::ScopedProfileRange spr("linear", 2);
    // Input x is [N, T, C], contiguity optional
    auto scores = linear(x);
    if (activation) {
        scores = activation(scores) * scale;
    }

    // Output is [N, T, C], contiguous
    return scores;
}

#if DORADO_CUDA_BUILD
void LinearCRFImpl::reserve_working_memory(WorkingMemory &wm) {
    bool use_torch = utils::get_dev_opt<bool>("torch_linear", false) || !koi_can_use_cutlass();
    if (use_torch && wm.layout != TensorLayout::NTC) {
        wm.next_TC(wm.T, wm.C, TensorLayout::NTC);
    }
    wm.next_TC(wm.T, int(linear->weight.size(0)), TensorLayout::NTC);
}
void LinearCRFImpl::run_koi(WorkingMemory &wm) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto type_id = (wm.layout == TensorLayout::CUTLASS_TNC_I8) ? KOI_I8 : KOI_F16;
    int C_in = wm.C;
    int C_out = int(linear->weight.size(0));
    void *bias_ptr = bias ? linear->bias.data_ptr() : nullptr;

    bool use_torch = utils::get_dev_opt<bool>("torch_linear", false) || !koi_can_use_cutlass();
    if (use_torch || (wm.layout == TensorLayout::NTC && !activation)) {
        utils::ScopedProfileRange spr("linear", 2);
        if (wm.layout != TensorLayout::NTC) {
            // Convert/transpose input layout to NTC, F16 if necessary
            utils::ScopedProfileRange spr_convert("convert_to_f16_ntc", 3);
            auto in = wm.get_current_NTC_view();
            auto out = wm.next_TC(wm.T, wm.C, TensorLayout::NTC);
            host_convert(stream, in.data_ptr(), int(in.stride(0)), int(in.stride(1)),
                         int(in.stride(2)), type_id, out.data_ptr(), int(out.stride(0)),
                         int(out.stride(1)), int(out.stride(2)), KOI_F16, int(in.size(0)),
                         int(in.size(1)), int(in.size(2)));
        }

        auto in = wm.current;
        auto out = wm.next_TC(wm.T, C_out, TensorLayout::NTC);
        auto out_2D = out.view({-1, C_out});
        if (!w_device.defined()) {
            w_device = linear->weight.t().contiguous().to(in.device());
        }
        dorado::utils::matmul_f16(in.view({-1, C_in}), w_device, out_2D);
        if (activation) {
            host_bias_activation_f16_inplace(stream, wm.T * wm.N, C_out, C_out, out_2D.data_ptr(),
                                             bias_ptr, KOI_TANH_X5);
        } else if (bias) {
            out_2D += linear->bias;
        }
    } else {
#if DORADO_TX2  // Koi for TX2 does not have Cutlass kernels
        throw std::logic_error("No Cutlass kernels in Jetson TX2 build.");
#else
        utils::ScopedProfileRange spr("koi_linear", 2);
        auto in_ntc = wm.get_current_NTC_view();
        auto out = wm.next_TC(wm.T, C_out, TensorLayout::NTC);
        if (!w_device.defined()) {
            if (type_id == KOI_F16) {
                w_device = linear->weight.contiguous().to(in_ntc.options());
            } else {
                auto scaled_tensor = dorado::utils::quantize_tensor(linear->weight, 1);
                weight_scale = scaled_tensor.scale.to(torch::kF16).to(in_ntc.device());
                w_device = scaled_tensor.t.contiguous().to(in_ntc.device());
            }
        }
        auto res = host_linear(
                stream, type_id, activation ? KOI_TANH_X5 : KOI_IDENTITY, KOI_F16, wm.N, wm.T, C_in,
                C_out, int(in_ntc.stride(0)), int(in_ntc.stride(1)), int(out.stride(0)),
                int(out.stride(1)), in_ntc.data_ptr(), w_device.data_ptr(), out.data_ptr(),
                weight_scale.defined() ? weight_scale.data_ptr() : nullptr, bias_ptr);
        if (res != KOI_SUCCESS) {
            throw std::runtime_error(std::string("Linear layer error:") + std::to_string(res));
        }
#endif  // ifdef DORADO_TX2 else
    }
}
#endif

ClampImpl::ClampImpl(float _min, float _max, bool _active)
        : active(_active), min(_min), max(_max) {}

at::Tensor ClampImpl::forward(at::Tensor x) {
    if (active) {
        utils::ScopedProfileRange spr("clamp", 2);
        x.clamp_(min, max);
    }
    return x;
}
}  // namespace dorado::nn