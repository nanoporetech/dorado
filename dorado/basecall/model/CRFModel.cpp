#include "CRFModel.h"

#include "config/BasecallModelConfig.h"
#include "nn/FLSTMStack.h"
#include "nn/FactorisedLinearLayer.h"
#include "nn/LSTMStack.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"

#include <optional>

#if DORADO_CUDA_BUILD
#include "nn/KoiUtils.h"

#include <c10/cuda/CUDAGuard.h>
#endif

using Slice = torch::indexing::Slice;

namespace dorado::basecall::model {

using namespace dorado::nn;
using namespace dorado::config;

CRFModelImpl::CRFModelImpl(const BasecallModelConfig &config, at::TensorOptions opts) {
    const auto cv = config.convs;
    const auto lstm_size = config.lstm_size;
    const bool tanh_x5 = config.scale == 5.f;
    convs = register_module("convs", ConvStack(cv));
    if (config.is_flstm_model()) {
        rnns = std::static_pointer_cast<RNNStackImpl>(
                register_module("rnns", FLSTMStack(config.lstm_layers, lstm_size,
                                                   config.lstm_inner_dim.value(), true, opts)));
    } else {
        rnns = std::static_pointer_cast<RNNStackImpl>(
                register_module("rnns", LSTMStack(config.lstm_layers, lstm_size, true)));
    }

    if (config.out_features.has_value()) {
#if DORADO_CUDA_BUILD
        const bool can_use_koi = opts.device().is_cuda() && koi_can_run_flstm();
#else
        const bool can_use_koi = false;
#endif

        // The linear layer is decomposed into 2 matmuls.
        const int decomposition = config.out_features.value();
        if (can_use_koi && (lstm_size == 1024) && (decomposition == 128) && tanh_x5) {
            linear1 = std::static_pointer_cast<LinearLayerImpl>(register_module(
                    "factorised_linear",
                    FactorisedLinearLayer(lstm_size, decomposition, config.outsize)));
            clamp1 = Clamp(-5.0, 5.0, config.clamp);
        } else {
            linear1 = std::static_pointer_cast<LinearLayerImpl>(register_module(
                    "linear1", LinearCRF(lstm_size, decomposition, config.bias, false)));
            linear2 = register_module("linear2",
                                      LinearCRF(decomposition, config.outsize, false, tanh_x5));
            clamp1 = Clamp(-5.0, 5.0, config.clamp);
        }
    } else if ((config.convs[0].size > 4) && (config.num_features == 1)) {
        // v4.x model without linear decomposition
        linear1 = std::static_pointer_cast<LinearLayerImpl>(
                register_module("linear1", LinearCRF(lstm_size, config.outsize, false, tanh_x5)));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
    } else {
        // Pre v4 model
        linear1 = std::static_pointer_cast<LinearLayerImpl>(
                register_module("linear1", LinearCRF(lstm_size, config.outsize, true, true)));
    }
}

void CRFModelImpl::load_state_dict(const std::vector<at::Tensor> &weights) {
    utils::load_state_dict(*this, weights);
}

#if DORADO_CUDA_BUILD
at::Tensor CRFModelImpl::run_koi(const at::Tensor &in,
                                 const nn::AuxiliaryData *const aux,
                                 at::Tensor *const signal_embeddings) {
    c10::cuda::CUDAGuard device_guard(in.device());
    WorkingMemory wm;
    {
        utils::ScopedProfileRange spr("reserve_working_memory", 2);
        // Input is [N, C, T] -- TODO: change to [N, T, C] on the input buffer side?

        // Determine working memory size
        if (aux) {
            wm.next_NTC(1, int(aux->NT_in_max()), int(in.size(1)), TensorLayout::NTC);
        } else {
            wm.next_NTC(int(in.size(0)), int(in.size(2)), int(in.size(1)), TensorLayout::NTC);
        }

        convs->reserve_working_memory(wm, aux, std::nullopt);
        rnns->reserve_working_memory(wm);
        linear1->reserve_working_memory(wm, aux);
        if (linear2) {
            linear2->reserve_working_memory(wm, nullptr);
        }

        wm.allocate_backing_tensor(in.device());

        // Copy `in` to working memory and run the model
        if (aux) {
            auto wm_in = wm.next_NTC(1, int(aux->NT_in()), int(in.size(1)), TensorLayout::NTC);
            wm_in.index({Slice()}) = in.transpose(1, 2).narrow(1, 0, aux->NT_in());
        } else {
            auto wm_in = wm.next_NTC(-1, int(in.size(2)), int(in.size(1)), TensorLayout::NTC);
            wm_in.index({Slice()}) = in.transpose(1, 2);
        }
    }

    convs->run_koi(wm, aux);
    rnns->run_koi(wm, aux);
    if (signal_embeddings) {
        auto embeddings = wm.get_current_NTC_view();
        // Koi's quantised recurrent kernels use a fixed int8 scale of 127. Convert back to the
        // floating-point representation expected by the latent-space projection.
        *signal_embeddings = embeddings.scalar_type() == torch::kInt8
                                     ? embeddings.to(torch::kFloat16).div_(127.f)
                                     : embeddings.clone();
    }
    linear1->run_koi(wm, aux);
    if (linear2) {
        linear2->run_koi(wm, nullptr);
    }

    // Clamping the scores to [-5, 5], if active (i.e. the role of `clamp1`), is performed by
    // `CUDADecoder` on reading the scores. This eliminates the cost of a large matrix
    // read-modify-write operation.

    // Output is [N, T, C], F16 or I8, contiguous
    assert((wm.layout == TensorLayout::NTC) || (wm.layout == TensorLayout::NTC_I8));
    return wm.current;
}
#endif

at::Tensor CRFModelImpl::forward(const at::Tensor &x,
                                 [[maybe_unused]] nn::AuxiliaryData *const aux,
                                 at::Tensor *const signal_embeddings) {
    utils::ScopedProfileRange spr("nn_forward", 1);
#if DORADO_CUDA_BUILD
    if (x.is_cuda() && x.dtype() == torch::kF16) {
        // Output is [N, T, C]
        return run_koi(x, aux, signal_embeddings);
    }
#endif
    auto h = convs->forward(x);
    h = rnns->forward(h);
    if (signal_embeddings) {
        *signal_embeddings = h;
    }
    h = linear1->forward(h);
    if (linear2) {
        h = linear2->forward(h);
    }
    if (clamp1) {
        h = clamp1->forward(h);
    }
    // Output is [N, T, C].
    return h;
}

}  // namespace dorado::basecall::model
