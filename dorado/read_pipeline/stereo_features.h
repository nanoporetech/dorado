#pragma once

#include "ReadPipeline.h"

#include <ATen/ATen.h>

namespace dorado {

// Generates the stereo duplex feature tensor from the supplied inputs.
at::Tensor GenerateStereoFeatures(const DuplexRead::StereoFeatureInputs& feature_inputs);

}  // namespace dorado