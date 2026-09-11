#pragma once

#include <cmath>
#include <string>
#include <utility>
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
    FLSTM_SOFTOUT,
    LINEAR,
    LINEAR_CRF_ENCODER,
    LSTM,
    PERMUTE,
    TRANSFORMER_ENCODER,
    UPSAMPLE,
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
    int inner_dim;
    std::string to_string() const;
};

ConvParams parse_conv_params(const toml::value& segment, bool has_clamp_next);
std::vector<ConvParams> parse_convs(const std::vector<toml::value>& sublayers);

struct TransformerEncoderLayerParams {
    // The number of expected features in the encoder/decoder inputs.
    int d_model{-1};
    // The number of heads in the multi-head attention model.
    int nhead{-1};
    // The dimension of the feedforward model.
    int dim_feedforward{-1};
    // Pair defining the possibly asymmetric sliding attention window mask.
    std::pair<int, int> attn_window{-1, -1};
    // DeepNorm normalization parameters.
    float deepnorm_alpha{1.0f};
    float deepnorm_beta{1.0f};
    // Multi-head attention RoPE parameters.
    float theta{10000.0f};
    int max_seq_len{0};

    std::string to_string() const;
};

TransformerEncoderLayerParams parse_transformer_encoder_layer_params(const toml::value& layer);

struct LinearUpsampleParams {
    int size;             // Number of input features
    int scale_factor{1};  // Linear upsample scale factor
    std::string to_string() const;
};

}  // namespace dorado::config
