#pragma once

#include <toml_fwd.hpp>

#include <cmath>
#include <string>

namespace dorado::config {

// Activation functions
enum class Activation : std::uint8_t { SWISH, SWISH_CLAMP, TANH };
std::string to_string(const Activation& activation);

// Model config encoder.sublayers.type variants
enum class SublayerType : std::uint8_t {
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