#pragma once

#include "read_pipeline/base/ReadPipeline.h"

#include <ATen/Tensor.h>

namespace dorado {

// Generates the stereo duplex feature tensor from the supplied inputs.
at::Tensor generate_stereo_features(const DuplexRead::StereoFeatureInputs& feature_inputs);

}  // namespace dorado