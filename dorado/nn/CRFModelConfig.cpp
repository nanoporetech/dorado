#include "CRFModelConfig.h"

#include "models/models.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <toml/value.hpp>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

enum SublayerType { CLAMP, CONVOLUTION, LINEAR, LINEAR_CRF_ENCODER, LSTM, PERMUTE, UNRECOGNISED };
static const std::unordered_map<std::string, SublayerType> sublayer_map = {
        {"clamp", SublayerType::CLAMP},   {"convolution", SublayerType::CONVOLUTION},
        {"linear", SublayerType::LINEAR}, {"linearcrfencoder", SublayerType::LINEAR_CRF_ENCODER},
        {"lstm", SublayerType::LSTM},     {"permute", SublayerType::PERMUTE},
};

// Parse encoder.sublayers.type attribute
SublayerType sublayer_type(const toml::value &segment) {
    const auto type = toml::find<std::string>(segment, "type");
    auto mapping_iter = sublayer_map.find(type);
    if (mapping_iter == sublayer_map.end()) {
        return SublayerType::UNRECOGNISED;
    }
    return mapping_iter->second;
}

namespace dorado {

// Parse the config to determine if there are any clamp layers
bool has_clamp(const std::vector<toml::value> &sublayers) {
    for (const auto &segment : sublayers) {
        if (sublayer_type(segment) == SublayerType::CLAMP) {
            return true;
        }
    }
    return false;
}

// Parse sublayer extracting convolution parameters. This is for use on v4+ models only
ConvParams parse_conv_params(const toml::value &segment, bool clamp) {
    ConvParams params;
    params.insize = toml::find<int>(segment, "insize");
    params.size = toml::find<int>(segment, "size");
    params.winlen = toml::find<int>(segment, "winlen");
    params.stride = toml::find<int>(segment, "stride");

    const auto &activation = toml::find<std::string>(segment, "activation");
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
std::vector<ConvParams> parse_convs(const std::vector<toml::value> &sublayers, bool clamp) {
    std::vector<ConvParams> convs;
    for (const auto &segment : sublayers) {
        if (sublayer_type(segment) == SublayerType::CONVOLUTION) {
            ConvParams conv = parse_conv_params(segment, clamp);
            convs.push_back(conv);
        }
    }
    return convs;
}

// Parse a the config.toml to resolve the scaling parameters.
SignalNormalisationParams parse_signal_normalisation_params(const toml::value &config_toml,
                                                            const std::string &model_name) {
    SignalNormalisationParams params;

    // med_mad scaling set based on filename for r9.4.1 models (~v3)
    if (model_name.rfind("dna_r9.4.1", 0) == 0) {
        params.strategy = ScalingStrategy::MED_MAD;
    }

    // scaling.strategy introduced with v4.3 models
    if (config_toml.contains("scaling")) {
        const auto &scaling = toml::find(config_toml, "scaling");
        params.strategy =
                scaling_strategy_from_string(toml::find<std::string>(scaling, "strategy"));
    }

    if (config_toml.contains("normalisation")) {
        const auto &norm = toml::find(config_toml, "normalisation");
        params.quantile_a = toml::find<float>(norm, "quantile_a");
        params.quantile_b = toml::find<float>(norm, "quantile_b");
        params.shift_multiplier = toml::find<float>(norm, "shift_multiplier");
        params.scale_multiplier = toml::find<float>(norm, "scale_multiplier");

        if (params.strategy != ScalingStrategy::QUANTILE) {
            spdlog::warn(
                    "Normalisation parameters are only used when `scaling.strategy = quantile`");
        }
    }

    return params;
}

// Check all encoder sublayers for unrecognised types and warn if any
void warn_unrecognised_sublayers(const std::vector<toml::value> &sublayers) {
    std::set<std::string> unique;
    for (const auto &segment : sublayers) {
        if (sublayer_type(segment) == SublayerType::UNRECOGNISED) {
            const auto type = toml::find<std::string>(segment, "type");
            if (unique.count(type) == 0) {
                spdlog::warn("Unrecognised sublayer type: `{}`", type);
                unique.insert(type);
            }
        }
    }
}

SampleType get_model_type(const std::string &model_name) {
    if (model_name.find("rna004") != std::string::npos) {
        return SampleType::RNA004;
    } else if (model_name.find("rna002") != std::string::npos) {
        return SampleType::RNA002;
    } else if (model_name.find("dna") != std::string::npos) {
        return SampleType::DNA;
    } else {
        throw std::runtime_error("Could not determine model type for " + model_name);
    }
}

std::string SignalNormalisationParams::to_string() const {
    std::string str = "SignalNormalisationParams {";
    str += " strategy:" + dorado::to_string(strategy);
    if (strategy == ScalingStrategy::QUANTILE) {
        str += " quantile_a:" + std::to_string(quantile_a);
        str += " quantile_b:" + std::to_string(quantile_b);
        str += " shift_multiplier:" + std::to_string(shift_multiplier);
        str += " scale_multiplier:" + std::to_string(scale_multiplier);
    }
    str += "}";
    return str;
}

std::string ConvParams::to_string() const {
    std::string str = "ConvParams {";
    str += " insize:" + std::to_string(insize);
    str += " size:" + std::to_string(size);
    str += " winlen:" + std::to_string(winlen);
    str += " stride:" + std::to_string(stride);
    str += " activation:" + dorado::to_string(activation);
    str += "}";
    return str;
};

std::string CRFModelConfig::to_string() const {
    std::string str = "CRFModelConfig {";
    str += " qscale:" + std::to_string(qscale);
    str += " qbias:" + std::to_string(qbias);
    str += " stride:" + std::to_string(stride);
    str += " bias:" + std::to_string(bias);
    str += " clamp:" + std::to_string(clamp);
    str += " out_features:" + std::to_string(out_features.value_or(-1));
    str += " state_len:" + std::to_string(state_len);
    str += " outsize:" + std::to_string(outsize);
    str += " blank_score:" + std::to_string(blank_score);
    str += " scale:" + std::to_string(scale);
    str += " num_features:" + std::to_string(num_features);
    str += " sample_rate:" + std::to_string(sample_rate);
    str += " mean_qscore_start_pos:" + std::to_string(mean_qscore_start_pos);
    str += " signal_norm_params:" + signal_norm_params.to_string();
    str += " convs: {";
    for (size_t c = 0; c < convs.size(); c++) {
        str += " " + std::to_string(c) + ": " + convs[c].to_string();
    }
    str += "}";
    return str;
};

CRFModelConfig load_crf_model_config(const std::filesystem::path &path) {
    const toml::value config_toml = toml::parse(path / "config.toml");

    CRFModelConfig config;
    config.model_path = path;

    if (config_toml.contains("qscore")) {
        const auto &qscore = toml::find(config_toml, "qscore");
        config.qbias = toml::find<float>(qscore, "bias");
        config.qscale = toml::find<float>(qscore, "scale");
        if (qscore.contains("mean_qscore_start_pos")) {
            config.mean_qscore_start_pos = toml::find<int32_t>(qscore, "mean_qscore_start_pos");
        }
    } else {
        spdlog::debug("> no qscore calibration found");
    }

    const auto &input = toml::find(config_toml, "input");
    config.num_features = toml::find<int>(input, "features");

    const auto &encoder = toml::find(config_toml, "encoder");
    if (encoder.contains("type")) {
        const std::vector<toml::value> sublayers =
                toml::find(config_toml, "encoder", "sublayers").as_array();

        warn_unrecognised_sublayers(sublayers);
        config.bias = false;

        // v4-type model
        config.clamp = has_clamp(sublayers);
        config.convs = parse_convs(sublayers, config.clamp);
        // Overall stride is the product of all conv layers' strides.
        for (const auto cv : config.convs) {
            config.stride *= cv.stride;
        }
        config.lstm_size = config.convs.back().size;

        for (const auto &segment : sublayers) {
            const auto type = sublayer_type(segment);
            if (type == SublayerType::LINEAR) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                config.out_features = toml::find<int>(segment, "out_features");
                config.bias = config.lstm_size > 128;
            } else if (type == SublayerType::LINEAR_CRF_ENCODER) {
                config.blank_score = toml::find<float>(segment, "blank_score");
            }
        }
    } else {
        // pre-v4 model
        config.stride = toml::find<int>(encoder, "stride");
        config.lstm_size = toml::find<int>(encoder, "features");
        config.blank_score = toml::find<float>(encoder, "blank_score");
        config.scale = toml::find<float>(encoder, "scale");

        const int first_conv = encoder.contains("first_conv_size")
                                       ? toml::find<int>(encoder, "first_conv_size")
                                       : 4;

        // pre-v4 config.clamp = false so Activation::SWISH not Activation::SWISH_CLAMP
        config.convs.push_back(
                ConvParams{config.num_features, first_conv, 5, 1, Activation::SWISH});
        config.convs.push_back(ConvParams{first_conv, 16, 5, 1, Activation::SWISH});
        config.convs.push_back(
                ConvParams{16, config.lstm_size, 19, config.stride, Activation::SWISH});
    }

    const auto &global_norm = toml::find(config_toml, "global_norm");
    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    config.state_len = toml::find<int>(global_norm, "state_len");

    // All of the paths avoid outputting explicit stay scores from the NN,
    // so we have 4^bases * 4 transitions.
    const auto PowerOf4 = [](int x) { return 1 << (x << 1); };
    config.outsize = PowerOf4(config.state_len + 1);

    // Fetch run_info parameters.
    // Do nothing if run_info is not available in config file.
    if (config_toml.contains("run_info")) {
        const auto &run_info = toml::find(config_toml, "run_info");
        config.sample_rate = toml::find<int>(run_info, "sample_rate");
    }

    std::string model_name = std::filesystem::canonical(config.model_path).filename().string();
    config.signal_norm_params = parse_signal_normalisation_params(config_toml, model_name);

    if (config.convs.size() != 3) {
        throw std::runtime_error("Expected 3 convolution layers but found: " +
                                 std::to_string(config.convs.size()));
    }
    if (config.convs[0].size != 4 && config.convs[0].size != 16) {
        throw std::runtime_error(
                "Invalid CRF model configuration - first convolution layer must be size 4 or 16. "
                "Got: " +
                std::to_string(config.convs[0].size));
    }

    config.sample_type = get_model_type(model_name);

    return config;
}

uint16_t get_model_sample_rate(const std::filesystem::path &model_path) {
    std::string model_name = std::filesystem::canonical(model_path).filename().string();
    // Find the sample rate from model config.
    int model_sample_rate = load_crf_model_config(model_path).sample_rate;
    if (model_sample_rate < 0) {
        // If unsuccessful, find sample rate by model name.
        model_sample_rate = models::get_sample_rate_by_model_name(model_name);
    }
    return model_sample_rate;
}

int32_t get_model_mean_qscore_start_pos(const CRFModelConfig &model_config) {
    int32_t mean_qscore_start_pos = model_config.mean_qscore_start_pos;
    if (mean_qscore_start_pos < 0) {
        // If unsuccessful, find start position by model name.
        std::string model_name = model_config.model_path.filename().string();
        mean_qscore_start_pos = models::get_mean_qscore_start_pos_by_model_name(model_name);
    }
    if (mean_qscore_start_pos < 0) {
        throw std::runtime_error("Mean q-score start position cannot be < 0");
    }
    return mean_qscore_start_pos;
}

bool is_rna_model(const CRFModelConfig &model_config) {
    auto path = std::filesystem::canonical(model_config.model_path);
    auto filename = path.filename();
    return filename.u8string().rfind("rna", 0) == 0;
}

std::string to_string(const Activation &activation) {
    switch (activation) {
    case Activation::SWISH:
        return std::string("swish");
    case Activation::SWISH_CLAMP:
        return std::string("swish_clamp");
    case Activation::TANH:
        return std::string("tanh");
    default:
        return std::string("UNKNOWN");
    };
}

std::string to_string(const ScalingStrategy &strategy) {
    switch (strategy) {
    case ScalingStrategy::MED_MAD:
        return std::string("med_mad");
    case ScalingStrategy::QUANTILE:
        return std::string("quantile");
    case ScalingStrategy::PA:
        return std::string("pa");
    default:
        throw std::runtime_error("Unknown scaling strategy");
    };
}

ScalingStrategy scaling_strategy_from_string(const std::string &strategy) {
    if (strategy == "med_mad") {
        return ScalingStrategy::MED_MAD;
    }
    if (strategy == "quantile") {
        return ScalingStrategy::QUANTILE;
    }
    if (strategy == "pa") {
        return ScalingStrategy::PA;
    }
    throw std::runtime_error("Unknown scaling strategy: `" + strategy + "`");
}

}  // namespace dorado
