#include "config/BasecallModelConfig.h"

#include "models/kits.h"
#include "models/models.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <cstddef>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// the mean Q-score of short reads are artificially lowered because of
// some lower quality bases at the beginning of the read. to correct for
// that, mean Q-score calculation should ignore the first few bases. The
// number of bases to ignore is dependent on the model.
uint32_t get_mean_qscore_start_pos_by_model_name(const std::string &model_name) {
    static const std::unordered_map<std::string, uint16_t> mean_qscore_start_pos_by_model = {
            // To add model specific start positions for older models,
            // create an entry keyed by model name with the value as
            // the desired start position.
            // e.g. {"dna_r10.4.1_e8.2_5khz_400bps_fast@v4.2.0", 10}
    };

    auto iter = mean_qscore_start_pos_by_model.find(model_name);
    if (iter != mean_qscore_start_pos_by_model.end()) {
        return iter->second;
    } else {
        // Assume start position of 60 as default.
        return 60;
    }
}

}  // namespace

namespace keys {
namespace {
// Workaround GCC-13 dangling reference warnings by passing an lvalue instead of a temporary
const std::string QSCORE{"qscore"};
const std::string POLYA{"poly_a"};
const std::string CALIBRATION_COEFFS{"calibration_coefficients"};
const std::string RUN_INFO{"run_info"};
const std::string SCALING{"scaling"};
const std::string NORM{"normalisation"};
const std::string INPUT{"input"};
const std::string GLOBAL_NORM{"global_norm"};
const std::string LAYER{"layer"};
const std::string SUBLAYERS{"sublayers"};
const std::string CONV{"conv"};
const std::string ENCODER{"encoder"};
}  // namespace
}  // namespace keys

namespace dorado::config {

namespace {

void parse_qscore_params(BasecallModelConfig &config, const toml::value &config_toml) {
    if (config_toml.contains(keys::QSCORE)) {
        const auto &qscore = toml::find(config_toml, keys::QSCORE);
        config.qbias = toml::find<float>(qscore, "bias");
        config.qscale = toml::find<float>(qscore, "scale");
        if (qscore.contains("mean_qscore_start_pos")) {
            config.mean_qscore_start_pos = toml::find<int32_t>(qscore, "mean_qscore_start_pos");
        } else {
            // If information is not present in the config, find start position by model name.
            const std::string model_name = config.model_path.filename().string();
            config.mean_qscore_start_pos = get_mean_qscore_start_pos_by_model_name(model_name);
        }
        if (config.mean_qscore_start_pos < 0) {
            throw std::runtime_error(
                    "model config error - qscore.mean_qscore_start_pos cannot be < 0");
        }
    } else {
        spdlog::debug("> no qscore calibration found");
    }
}

void parse_polya_coefficients(BasecallModelConfig &config, const toml::value &config_toml) {
    if (config_toml.contains(keys::POLYA)) {
        const auto &polya = toml::find(config_toml, keys::POLYA);
        if (polya.contains(keys::CALIBRATION_COEFFS)) {
            // handle old style models
            const auto &coeffs = toml::find(polya, keys::CALIBRATION_COEFFS);
            if (coeffs.is_array()) {
                const auto &coeffs_array = coeffs.as_array();
                if (std::size(coeffs_array) > 1) {
                    spdlog::warn(
                            "'polya.calibration_coefficients' does not support multiple values. "
                            "Discarding higher order coefficients.");
                }
                config.polya_speed_correction =
                        1.f / static_cast<float>(coeffs_array[0].as_floating());
            } else if (coeffs.is_floating() || coeffs.is_integer()) {
                config.polya_speed_correction = 1.f / static_cast<float>(coeffs.as_floating());
            } else {
                throw std::runtime_error("Invalid type for polyA calibration coefficients in " +
                                         config.model_path.string());
            }
        } else {
            if (polya.contains("speed_correction") || polya.contains("offset_correction")) {
                if (!(polya.contains("speed_correction") && polya.contains("offset_correction"))) {
                    throw std::runtime_error(
                            "model config error - must contain both 'polya.speed_correction' and "
                            "'polya.offset_correction' or neither.");
                }
                config.polya_speed_correction = toml::find<float>(polya, "speed_correction");
                config.polya_offset_correction = toml::find<float>(polya, "offset_correction");
            }
        }
    }
}

void parse_run_info(BasecallModelConfig &config, const toml::value &config_toml) {
    // Fetch run_info parameters.
    // Do nothing if run_info is not available in config file.
    if (config_toml.contains(keys::RUN_INFO)) {
        const auto &run_info = toml::find(config_toml, keys::RUN_INFO);
        config.sample_rate = toml::find<int>(run_info, "sample_rate");

        if (run_info.contains("sample_type")) {
            config.sample_type =
                    models::get_sample_type(toml::find<std::string>(run_info, "sample_type"));
        }
    }

    using namespace dorado::models;
    if (config.sample_type == SampleType::UNKNOWN) {
        const std::string model_name =
                std::filesystem::canonical(config.model_path).filename().string();
        config.sample_type = get_sample_type_from_model_name(model_name);
        if (config.sample_type == SampleType::UNKNOWN) {
            throw std::runtime_error(
                    "Failed to determine model sample type from model name or config");
        }
    }
}

// Parse the config to determine if there are any clamp layers
bool has_clamp(const std::vector<toml::value> &sublayers) {
    for (const auto &segment : sublayers) {
        if (sublayer_type(toml::find<std::string>(segment, "type")) == SublayerType::CLAMP) {
            return true;
        }
    }
    return false;
}

// Parse a the config.toml to resolve the scaling parameters.
SignalNormalisationParams parse_signal_normalisation_params(const toml::value &config_toml) {
    SignalNormalisationParams params;

    // scaling.strategy introduced with v4.3 models
    if (config_toml.contains(keys::SCALING)) {
        const auto &scaling = toml::find(config_toml, keys::SCALING);
        params.strategy =
                scaling_strategy_from_string(toml::find<std::string>(scaling, "strategy"));
    }

    if (config_toml.contains(keys::NORM)) {
        const auto &norm = toml::find(config_toml, keys::NORM);
        params.quantile.quantile_a = toml::find<float>(norm, "quantile_a");
        params.quantile.quantile_b = toml::find<float>(norm, "quantile_b");
        params.quantile.shift_multiplier = toml::find<float>(norm, "shift_multiplier");
        params.quantile.scale_multiplier = toml::find<float>(norm, "scale_multiplier");

        if (params.strategy != ScalingStrategy::QUANTILE) {
            spdlog::warn(
                    "Normalisation parameters are only used when `scaling.strategy = quantile`");
        }
    }

    if (config_toml.contains("standardisation")) {
        const auto norm = toml::find(config_toml, "standardisation");
        params.standardisation.standardise = toml::find<int>(norm, "standardise") > 0;
        if (params.standardisation.standardise) {
            params.standardisation.mean = toml::find<float>(norm, "mean");
            params.standardisation.stdev = toml::find<float>(norm, "stdev");
        }

        if (params.standardisation.standardise && params.strategy != ScalingStrategy::PA) {
            throw std::runtime_error(
                    "Signal standardisation is implemented only for `scaling.strategy = pa`");
        }

        if (params.standardisation.stdev <= 0.0f) {
            throw std::runtime_error(
                    "Config error: `standardisation.stdev` must be greater than 0, got: " +
                    std::to_string(params.standardisation.stdev));
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

BasecallModelConfig load_lstm_model_config(const std::filesystem::path &path) {
    const toml::value config_toml = toml::parse(path / "config.toml");

    BasecallModelConfig config;
    config.model_path = path;
    config.basecaller.update(path);

    parse_qscore_params(config, config_toml);
    parse_polya_coefficients(config, config_toml);

    const auto &input = toml::find(config_toml, keys::INPUT);
    config.num_features = toml::find<int>(input, "features");

    const auto &encoder = toml::find(config_toml, keys::ENCODER);
    if (encoder.contains("type")) {
        const std::vector<toml::value> sublayers =
                toml::find(config_toml, keys::ENCODER, "sublayers").as_array();

        warn_unrecognised_sublayers(sublayers);
        config.bias = false;

        // v4-type model
        config.clamp = has_clamp(sublayers);
        config.convs = parse_convs(sublayers);
        // Overall stride is the product of all conv layers' strides.
        for (const auto &cv : config.convs) {
            config.stride *= cv.stride;
        }
        config.lstm_size = config.convs.back().size;
        config.lstm_layers = 0;  // Count the number of lstm sublayers
        for (const auto &segment : sublayers) {
            const auto type = sublayer_type(segment);
            if (type == SublayerType::LINEAR) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                config.out_features = toml::find<int>(segment, "out_features");
                config.bias = config.lstm_size > 128;
            } else if (type == SublayerType::LINEAR_CRF_ENCODER) {
                config.blank_score = toml::find<float>(segment, "blank_score");
            } else if (type == SublayerType::LSTM) {
                config.lstm_layers++;
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

        config.convs.push_back(
                ConvParams{config.num_features, first_conv, 5, 1, Activation::SWISH});
        config.convs.push_back(ConvParams{first_conv, 16, 5, 1, Activation::SWISH});
        config.convs.push_back(
                ConvParams{16, config.lstm_size, 19, config.stride, Activation::SWISH});
    }

    const auto &global_norm = toml::find(config_toml, keys::GLOBAL_NORM);
    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    config.state_len = toml::find<int>(global_norm, "state_len");

    // All of the paths avoid outputting explicit stay scores from the NN,
    // so we have 4^bases * 4 transitions.
    const auto PowerOf4 = [](int x) { return 1 << (x << 1); };
    config.outsize = PowerOf4(config.state_len + 1);
    config.signal_norm_params = parse_signal_normalisation_params(config_toml);

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

    parse_run_info(config, config_toml);

    return config;
}

std::optional<toml::value> toml_get(const toml::value &value,
                                    const std::vector<std::string> &fields) {
    if (fields.empty()) {
        return std::nullopt;
    }
    const auto *v = &value;
    for (const auto &field : fields) {
        if (v->is_table() && v->contains(field)) {
            v = &v->as_table().at(field);
        } else {
            return std::nullopt;
        }
    }
    return *v;
}

}  // namespace

bool is_tx_model_config(const std::filesystem::path &path) {
    const auto config_toml = toml::parse(path / "config.toml");
    const auto res = toml_get(config_toml, {"model", "encoder", "transformer_encoder"});
    return res.has_value();
}

bool is_rna_model(const BasecallModelConfig &model_config) {
    switch (model_config.sample_type) {
    case models::SampleType::DNA:
        return false;
    case models::SampleType::RNA002:
        return true;
    case models::SampleType::RNA004:
        return true;
    case models::SampleType::UNKNOWN:
        throw std::logic_error("Model config sample type is unknown!");
    }
    // Exception to prevent us reaching the end of the function
    throw std::logic_error("Model config sample type is not recognised!");
}

bool is_duplex_model(const BasecallModelConfig &model_config) {
    return model_config.num_features > 1;
}

namespace {

TxEncoderParams parse_tx_encoder_params(const toml::value &cfg) {
    const auto &enc = toml::find(cfg, "model", "encoder", "transformer_encoder");
    TxEncoderParams params;
    params.depth = toml::find<int>(enc, "depth");

    const auto &layer = toml::find(enc, keys::LAYER);
    params.d_model = toml::find<int>(layer, "d_model");
    params.nhead = toml::find<int>(layer, "nhead");
    params.dim_feedforward = toml::find<int>(layer, "dim_feedforward");
    params.deepnorm_alpha = toml::find<float>(layer, "deepnorm_alpha");
    if (layer.contains("max_seq_len")) {
        params.max_seq_len = toml::find<int>(layer, "max_seq_len");
    }

    if (layer.contains("rotary_base") && layer.contains("theta")) {
        throw std::runtime_error(
                "Model Config Error. [model.encoder.transformer_encoder] 'rotary_base' and 'theta' "
                "are mutually exclusive.");
    } else if (layer.contains("theta")) {
        params.theta = toml::find<float>(layer, "theta");
    } else if (layer.contains("rotary_base")) {
        params.theta = toml::find<float>(layer, "rotary_base");
    }

    const auto attn_window_ = toml::find(layer, "attn_window").as_array();
    params.attn_window = {static_cast<int>(attn_window_[0].as_integer()),
                          static_cast<int>(attn_window_[1].as_integer())};
    return params;
}

LinearUpsampleParams parse_encoder_upsample_params(const toml::value &cfg) {
    const auto &ups = toml::find(cfg, "model", "encoder", "upsample");
    LinearUpsampleParams params;
    params.size = toml::find<int>(ups, "d_model");
    params.scale_factor = toml::find<int>(ups, "scale_factor");
    return params;
}

CRFEncoderParams parse_crf_encoder_params(const toml::value &cfg) {
    const auto &crf = toml::find(cfg, "model", "encoder", "crf");
    CRFEncoderParams params;
    params.insize = toml::find<int>(crf, "insize");
    params.n_base = toml::find<int>(crf, "n_base");
    params.state_len = toml::find<int>(crf, "state_len");
    params.scale = toml::find<float>(crf, "scale");
    params.blank_score = toml::find<float>(crf, "blank_score");
    params.expand_blanks = toml::find<bool>(crf, "expand_blanks");
    params.permute = toml::find<std::vector<int>>(crf, "permute");

    return params;
}

BasecallModelConfig load_tx_model_config(const std::filesystem::path &path) {
    const auto config_toml = toml::parse(path / "config.toml");
    const auto model_toml = toml::find(config_toml, "model");

    BasecallModelConfig config;

    config.model_path = path;
    config.basecaller.update(path);

    parse_qscore_params(config, config_toml);

    const TxEncoderParams tx_encoder = parse_tx_encoder_params(config_toml);
    const LinearUpsampleParams upsample = parse_encoder_upsample_params(config_toml);
    const CRFEncoderParams crf_encoder = parse_crf_encoder_params(config_toml);

    config.tx = TxStack{tx_encoder, upsample, crf_encoder};
    config.tx->check();

    const auto &convs = toml::find(model_toml, keys::ENCODER, keys::CONV);
    const auto &sublayers = toml::find(convs, keys::SUBLAYERS).as_array();
    for (const auto &segment : sublayers) {
        const auto type = toml::find<std::string>(segment, "type");
        if (type.compare("convolution") != 0) {
            continue;
        }

        const auto conv = parse_conv_params(segment, false /* Tx models do not have swish clamp */);
        config.convs.push_back(conv);
        config.stride *= conv.stride;
    }
    // Recalculate the stride by accounting for upsampling / downsampling
    config.stride /= upsample.scale_factor;
    config.out_features = crf_encoder.out_features();
    config.outsize = crf_encoder.outsize();

    config.state_len = config.tx->crf.state_len;
    config.num_features = config.convs.front().insize;

    config.signal_norm_params = parse_signal_normalisation_params(config_toml);

    parse_run_info(config, config_toml);

    // Force downstream issue (negative lstm size) if a tx model config is incorrectly
    // used to define an LSTM model. Incorrect use should be guarded against by using is_tx_model()
    config.lstm_size = -1;

    return config;
}

}  // namespace

void TxStack::check() const {
    const auto eq = [](const int a, const int b, const std::string &msg) {
        if (a != b) {
            spdlog::warn("Transformer model params check - expected {} but {} != {}", msg, a, b);
        }
    };
    eq(crf.insize, tx.d_model, "linearcrfencoder.insize == transformer_encoder.layer.d_model");
    eq(upsample.size, tx.d_model, "linearupsample.d_model == transformer_encoder.layer.d_model");
}

bool BasecallModelConfig::has_normalised_basecaller_params() const {
    bool is_normalised = true;
    const auto cs = basecaller.chunk_size();
    const auto csg = chunk_size_granularity();
    if (cs % csg != 0) {
        spdlog::error("Expected normalised chunksize - got: {} - for granularity: {}", cs, csg);
        is_normalised = false;
    }

    const auto ov = basecaller.overlap();
    const auto si = stride_inner();
    if (ov % si != 0) {
        spdlog::error("Expected normalised overlap - got: {} - for model stride: {}", ov, si);
        is_normalised = false;
    }

    if (cs <= ov) {
        spdlog::error("Expected chunk size > overlap - got: {} - for overlap: {}", cs, ov);
        is_normalised = false;
    }
    return is_normalised;
}

BasecallModelConfig load_model_config(const std::filesystem::path &path) {
    const std::string model_name = std::filesystem::canonical(path).filename().string();
    models::throw_on_deprecated_model(model_name);
    return is_tx_model_config(path) ? load_tx_model_config(path) : load_lstm_model_config(path);
}

std::string to_string(const ScalingStrategy &strategy) {
    switch (strategy) {
    case ScalingStrategy::MED_MAD:
        return std::string("med_mad");
    case ScalingStrategy::QUANTILE:
        return std::string("quantile");
    case ScalingStrategy::PA:
        return std::string("pa");
    };
    throw std::runtime_error("Unknown scaling strategy");
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

std::string StandardisationScalingParams::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "StandardisationScalingParams {"
        << " standardise:" << standardise
        << " mean:"        << mean
        << " stdev:"       << stdev << "}";
    return oss.str();
    // clang-format on
}

std::string QuantileScalingParams::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "QuantileScalingParams {"
        << " quantile_a:"       << quantile_a
        << " quantile_b:"       << quantile_b
        << " shift_multiplier:" << shift_multiplier
        << " scale_multiplier:" << scale_multiplier << "}";
    return oss.str();
    // clang-format on
}

std::string SignalNormalisationParams::to_string() const {
    std::ostringstream oss;
    oss << "SignalNormalisationParams {"
        << " strategy:" + config::to_string(strategy);
    if (strategy == ScalingStrategy::QUANTILE) {
        oss << " " + quantile.to_string();
    } else if (strategy == ScalingStrategy::PA && standardisation.standardise) {
        oss << " " + standardisation.to_string();
    }
    oss << " }";
    return oss.str();
}

std::string ConvParams::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "ConvParams {"
        << " insize:"   << insize
        << " size:"     << size
        << " winlen:"   << winlen
        << " stride:"   << stride
        << " activation:" << config::to_string(activation) << " }";
    return oss.str();
    // clang-format on
}

std::string TxEncoderParams::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "TxEncoderParams {"
        << " d_model:"          << d_model
        << " nhead:"            << nhead
        << " depth:"            << depth
        << " dim_feedforward:"  << dim_feedforward
        << " theta:"            << theta
        << " max_seq_len:"      << max_seq_len
        << " attn_window: ["    << attn_window.first << ", " << attn_window.second << "]"
        << " deepnorm_alpha:"   << deepnorm_alpha    << " }";
    return oss.str();
    // clang-format on
}

std::string CRFEncoderParams::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "CRFEncoderParams {"
        << " insize:"        << insize
        << " n_base:"        << n_base
        << " state_len:"     << state_len
        << " scale:"         << scale
        << " blank_score:"   << blank_score
        << " expand_blanks:" << expand_blanks
        << " permute:"       << std::boolalpha << !permute.empty() << " }";
    return oss.str();
    // clang-format on
}

std::string BasecallModelConfig::to_string() const {
    std::ostringstream oss;
    // clang-format off
    oss << "BasecallModelConfig {"
        << " qscale:"       << qscale
        << " qbias:"        << qbias
        << " stride:"       << stride
        << " bias:"         << bias
        << " clamp:"        << clamp
        << " out_features:" << out_features.value_or(-1)
        << " state_len:"    << state_len
        << " outsize:"      << outsize
        << " blank_score:"  << blank_score
        << " scale:"        << scale
        << " num_features:" << num_features
        << " sample_rate:"  << sample_rate
        << " sample_type:"  << models::get_sample_type_info(sample_type).name
        << " mean_qscore_start_pos:" << mean_qscore_start_pos
        << " " << signal_norm_params.to_string()
        << " " << basecaller.to_string();
    
    oss << " convs: {";
    for (size_t c = 0; c < convs.size(); c++) {
        oss << " " << c <<  ": " << convs[c].to_string();
    }
    oss << " }"; 

    if (is_lstm_model()) {
        oss << " model_type: lstm {";
    }
    if (is_tx_model()) {
        oss << " model_type: tx {" 
            << " crf_encoder: " << tx->crf.to_string()
            << " transformer: " << tx->tx.to_string()
            << " upsample: " << tx->upsample.to_string();
    }
    oss << "}}";
    return oss.str();
    // clang-format on
}

}  // namespace dorado::config
