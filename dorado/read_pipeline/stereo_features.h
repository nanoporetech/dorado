#pragma once

#include "ReadPipeline.h"

#include <ATen/ATen.h>

namespace dorado {

at::Tensor GenerateStereoFeatures(const DuplexRead::StereoFeatureInputs& feature_inputs);

}  // namespace dorado