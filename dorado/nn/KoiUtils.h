#pragma once

#if DORADO_CUDA_BUILD

namespace dorado::nn {

// TODO: These should really be part of Koi
bool koi_can_use_cutlass();
bool koi_can_use_quantised_lstm();

}  // namespace dorado::nn

#endif