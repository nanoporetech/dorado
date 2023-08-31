#include "CRFModelConfig.h"

#include "utils/models.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <string>

namespace dorado {

CRFModelConfig load_crf_model_config(const std::filesystem::path &path) {
    const auto config_toml = toml::parse(path / "config.toml");

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
        // v4-type model
        for (const auto &segment : toml::find(config_toml, "encoder", "sublayers").as_array()) {
            const auto type = toml::find<std::string>(segment, "type");
            if (type.compare("convolution") == 0) {
                // Overall stride is the product of all conv layers' strides.
                config.stride *= toml::find<int>(segment, "stride");
            } else if (type.compare("lstm") == 0) {
                config.insize = toml::find<int>(segment, "size");
            } else if (type.compare("linear") == 0) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                config.out_features = toml::find<int>(segment, "out_features");
            } else if (type.compare("clamp") == 0) {
                config.clamp = true;
            } else if (type.compare("linearcrfencoder") == 0) {
                config.blank_score = toml::find<float>(segment, "blank_score");
            }
        }
        config.conv = 16;
        config.bias = config.insize > 128;
    } else {
        // pre-v4 model
        config.stride = toml::find<int>(encoder, "stride");
        config.insize = toml::find<int>(encoder, "features");
        config.blank_score = toml::find<float>(encoder, "blank_score");
        config.scale = toml::find<float>(encoder, "scale");

        if (encoder.contains("first_conv_size")) {
            config.conv = toml::find<int>(encoder, "first_conv_size");
        }
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

    // Fetch signal normalisation parameters.
    // Use default values if normalisation section is not found.
    if (config_toml.contains("normalisation")) {
        const auto &norm = toml::find(config_toml, "normalisation");
        config.signal_norm_params.quantile_a = toml::find<float>(norm, "quantile_a");
        config.signal_norm_params.quantile_b = toml::find<float>(norm, "quantile_b");
        config.signal_norm_params.shift_multiplier = toml::find<float>(norm, "shift_multiplier");
        config.signal_norm_params.scale_multiplier = toml::find<float>(norm, "scale_multiplier");
    }

    // Set quantile scaling method based on the model filename
    std::string model_name = std::filesystem::canonical(config.model_path).filename().string();
    if (model_name.rfind("dna_r9.4.1", 0) == 0) {
        config.signal_norm_params.quantile_scaling = false;
    }

    return config;
}

uint16_t get_model_sample_rate(const std::filesystem::path &model_path) {
    std::string model_name = std::filesystem::canonical(model_path).filename().string();
    // Find the sample rate from model config.
    int model_sample_rate = load_crf_model_config(model_path).sample_rate;
    if (model_sample_rate < 0) {
        // If unsuccessful, find sample rate by model name.
        model_sample_rate = utils::get_sample_rate_by_model_name(model_name);
    }
    return model_sample_rate;
}

int32_t get_model_mean_qscore_start_pos(const CRFModelConfig &model_config) {
    int32_t mean_qscore_start_pos = model_config.mean_qscore_start_pos;
    if (mean_qscore_start_pos < 0) {
        // If unsuccessful, find start position by model name.
        std::string model_name = model_config.model_path.filename().string();
        mean_qscore_start_pos = utils::get_mean_qscore_start_pos_by_model_name(model_name);
    }
    if (mean_qscore_start_pos < 0) {
        throw std::runtime_error("Mean q-score start position cannot be < 0");
    }
    return mean_qscore_start_pos;
}

}  // namespace dorado
