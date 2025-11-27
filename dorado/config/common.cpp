#include "config/common.h"

#include <toml.hpp>

#include <stdexcept>
#include <unordered_map>

namespace keys {
namespace {
const std::string ACTIVATION{"activation"};
}
}  // namespace keys

namespace dorado::config {

static const std::unordered_map<std::string, SublayerType> sublayer_map = {
        {"clamp", SublayerType::CLAMP},
        {"convolution", SublayerType::CONVOLUTION},
        {"flstm", SublayerType::FLSTM},
        {"linear", SublayerType::LINEAR},
        {"linearcrfencoder", SublayerType::LINEAR_CRF_ENCODER},
        {"lstm", SublayerType::LSTM},
        {"permute", SublayerType::PERMUTE},
        {"upsample", SublayerType::UPSAMPLE},
};

std::string to_string(const Activation &activation) {
    switch (activation) {
    case Activation::SWISH:
        return std::string("swish");
    case Activation::SWISH_CLAMP:
        return std::string("swish_clamp");
    case Activation::TANH:
        return std::string("tanh");
    };
    throw std::runtime_error("Unknown activation function");
}

// Parse encoder.sublayers.type attribute
SublayerType sublayer_type(const toml::value &segment) {
    return sublayer_type(toml::find<std::string>(segment, "type"));
}

SublayerType sublayer_type(const std::string &type) {
    const auto mapping_iter = sublayer_map.find(type);
    if (mapping_iter == sublayer_map.cend()) {
        return SublayerType::UNRECOGNISED;
    }
    return mapping_iter->second;
}

// Parse sublayer extracting convolution parameters. This is for use on v4+ models only
ConvParams parse_conv_params(const toml::value &segment, const bool clamp) {
    ConvParams params;
    params.insize = toml::find<int>(segment, "insize");
    params.size = toml::find<int>(segment, "size");
    params.winlen = toml::find<int>(segment, "winlen");
    params.stride = toml::find<int>(segment, "stride");

    const auto &activation = toml::find<std::string>(segment, keys::ACTIVATION);
    if (activation == "swish") {
        params.activation = clamp ? Activation::SWISH_CLAMP : Activation::SWISH;
    } else if (activation == "tanh") {
        params.activation = Activation::TANH;
    } else {
        throw std::runtime_error("Unknown activation: `" + activation +
                                 "` in model config, expected `swish` or `tanh`");
    }

    return params;
}

// Parse sublayers extracting convolution parameters. This is for use on v4+ models only
std::vector<ConvParams> parse_convs(const std::vector<toml::value> &sublayers) {
    std::vector<ConvParams> convs;
    for (std::size_t i = 0; i < sublayers.size(); ++i) {
        // If the sublayer after a convolution is a clamp, the activation function may have
        // a fused implementation
        if (sublayer_type(sublayers.at(i)) == SublayerType::CONVOLUTION) {
            const bool has_clamp_next = ((i + 1) < sublayers.size()) &&
                                        (sublayer_type(sublayers.at(i + 1)) == SublayerType::CLAMP);
            ConvParams conv = parse_conv_params(sublayers.at(i), has_clamp_next);
            convs.push_back(conv);
        }
    }
    return convs;
}

std::string LinearUpsampleParams::to_string() const {
    std::ostringstream oss;
    oss << "LinearUpsampleParams {"
        << " size:" << size << " scale_factor:" << scale_factor << " }";
    return oss.str();
}

}  // namespace dorado::config