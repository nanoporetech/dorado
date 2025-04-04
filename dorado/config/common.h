#pragma once

#include <cmath>
#include <string>
#include <vector>

// Forward declaration of toml::value taken from toml_fwd.hpp
namespace toml {
template <typename TypeConfig>
class basic_value;

struct type_config;
using value = basic_value<type_config>;
}  // namespace toml

namespace dorado::config {

// Activation functions
enum class Activation { SWISH, SWISH_CLAMP, TANH };
std::string to_string(const Activation& activation);

// Model config encoder.sublayers.type variants
enum class SublayerType {
    CLAMP,
    CONVOLUTION,
    LINEAR,
    LINEAR_CRF_ENCODER,
    LSTM,
    PERMUTE,
    UNRECOGNISED
};

// Parse encoder.sublayers.type attribute
SublayerType sublayer_type(const toml::value& segment);
SublayerType sublayer_type(const std::string& type);

// Model config encoder.convolution parameters
struct ConvParams {
    int insize;
    int size;
    int winlen;
    int stride = 1;
    Activation activation;
    std::string to_string() const;
};

ConvParams parse_conv_params(const toml::value& segment, bool has_clamp_next);
std::vector<ConvParams> parse_convs(const std::vector<toml::value>& sublayers);

}  // namespace dorado::config