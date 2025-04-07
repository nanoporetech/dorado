#include "CRFModel.h"

#include "config/BasecallModelConfig.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/module_utils.h"
#include "torch_utils/tensor_utils.h"
#include "utils/math_utils.h"

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
#include "koi.h"
}
#endif

#include <torch/nn.h>

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;

namespace dorado::basecall::model {

using namespace dorado::nn;
using namespace dorado::config;

CRFModelImpl::CRFModelImpl(const BasecallModelConfig &config) {
    const auto cv = config.convs;
    const auto lstm_size = config.lstm_size;
    convs = register_module("convs", ConvStack(cv));
    rnns = register_module("rnns", LSTMStack(5, lstm_size));

    if (config.out_features.has_value()) {
        // The linear layer is decomposed into 2 matmuls.
        const int decomposition = config.out_features.value();
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, true, false));
        linear2 =
                register_module("linear2", LinearCRF(decomposition, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        encoder = Sequential(convs, rnns, linear1, linear2, clamp1);
    } else if ((config.convs[0].size > 4) && (config.num_features == 1)) {
        // v4.x model without linear decomposition
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        encoder = Sequential(convs, rnns, linear1, clamp1);
    } else {
        // Pre v4 model
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, true, true));
        encoder = Sequential(convs, rnns, linear1);
    }
}

void CRFModelImpl::load_state_dict(const std::vector<at::Tensor> &weights) {
    utils::load_state_dict(*this, weights);
}

#if DORADO_CUDA_BUILD
at::Tensor CRFModelImpl::run_koi(const at::Tensor &in) {
    // Input is [N, C, T] -- TODO: change to [N, T, C] on the input buffer side?
    c10::cuda::CUDAGuard device_guard(in.device());

    // Determine working memory size
    WorkingMemory wm(int(in.size(0)));
    wm.next_TC(int(in.size(2)), int(in.size(1)), TensorLayout::NTC);
    convs->reserve_working_memory(wm);
    rnns->reserve_working_memory(wm);
    linear1->reserve_working_memory(wm);
    if (linear2) {
        linear2->reserve_working_memory(wm);
    }

    wm.allocate_backing_tensor(in.device());

    // Copy `in` to working memory and run the model
    auto wm_in = wm.next_TC(int(in.size(2)), int(in.size(1)), TensorLayout::NTC);
    wm_in.index({Slice()}) = in.transpose(1, 2);

    convs->run_koi(wm);
    rnns->run_koi(wm);
    linear1->run_koi(wm);
    if (linear2) {
        linear2->run_koi(wm);
    }

    // Clamping the scores to [-5, 5], if active (i.e. the role of `clamp1`), is performed by
    // `CUDADecoder` on reading the scores. This eliminates the cost of a large matrix
    // read-modify-write operation.

    // Output is [N, T, C], F16, contiguous
    assert(wm.layout == TensorLayout::NTC);
    return wm.current;
}
#endif

at::Tensor CRFModelImpl::forward(const at::Tensor &x) {
    utils::ScopedProfileRange spr("nn_forward", 1);
#if DORADO_CUDA_BUILD
    if (x.is_cuda() && x.dtype() == torch::kF16) {
        // Output is [N, T, C]
        return run_koi(x);
    }
#endif
    // Output is [N, T, C]
    return encoder->forward(x);
}

}  // namespace dorado::basecall::model
