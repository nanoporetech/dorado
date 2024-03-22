#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dorado::basecall {

enum class Activation { SWISH, SWISH_CLAMP, TANH };
std::string to_string(const Activation& activation);

enum class ScalingStrategy { MED_MAD, QUANTILE, PA };
std::string to_string(const ScalingStrategy& strategy);
ScalingStrategy scaling_strategy_from_string(const std::string& strategy);

struct StandardisationScalingParams {
    bool standardise = false;
    float mean = 0.0f;
    float stdev = 1.0f;

    std::string to_string() const {
        std::string str = "StandardisationScalingParams {";
        str += " standardise:" + std::to_string(standardise);
        str += " mean:" + std::to_string(mean);
        str += " stdev:" + std::to_string(stdev);
        str += "}";
        return str;
    };
};

struct QuantileScalingParams {
    float quantile_a = 0.2f;
    float quantile_b = 0.9f;
    float shift_multiplier = 0.51f;
    float scale_multiplier = 0.53f;

    std::string to_string() const {
        std::string str = "QuantileScalingParams {";
        str += " quantile_a:" + std::to_string(quantile_a);
        str += " quantile_b:" + std::to_string(quantile_b);
        str += " shift_multiplier:" + std::to_string(shift_multiplier);
        str += " scale_multiplier:" + std::to_string(scale_multiplier);
        str += "}";
        return str;
    };
};

struct SignalNormalisationParams {
    ScalingStrategy strategy = ScalingStrategy::QUANTILE;

    QuantileScalingParams quantile;
    StandardisationScalingParams standarisation;

    std::string to_string() const;
};

struct ConvParams {
    int insize;
    int size;
    int winlen;
    int stride = 1;
    Activation activation;

    std::string to_string() const;
};

enum SampleType {
    DNA,
    RNA002,
    RNA004,
};

namespace tx {

struct TxEncoderParams {
    //  The number of expected features in the encoder/decoder inputs
    int d_model;
    // The number of heads in the multi-head attention (MHA) models
    int nhead;
    // The number of transformer layers
    int depth;
    // Linear upsample scale factor
    int scale_factor;

    // The dimension of the feedforward model
    int dim_feedforward() const { return int(d_model * (2.0f / 3.0f) / 32) * 32; }
    // The deepnorm normalisation alpha parameter
    float deepnorm_alpha() const { return float(pow(float(2 * depth), 0.25f)); }
    // The deepnorm normalisation beta constant
    float deepnorm_beta() const { return float(pow(float(8 * depth), -0.25f)); }

    std::string to_string() const;
};

struct BasecallerParams {
    int batchsize;
    int chunksize;
    int overlap;

    std::string to_string() const;
};

struct CRFEncoderParams {
    int output_stride;
    int n_base;
    int state_len;
    std::optional<float> blank_score;

    std::string to_string() const;
};

struct Params {
    BasecallerParams basecaller;
    TxEncoderParams tx;
    CRFEncoderParams crf;
};

}  // namespace tx

// Values extracted from config.toml used in construction of the model module.
struct CRFModelConfig {
    float qscale = 1.0f;
    float qbias = 0.0f;
    int lstm_size = 0;
    int stride = 1;
    bool bias = true;
    bool clamp = false;
    // If there is a decomposition of the linear layer, this is the bottleneck feature size.
    std::optional<int> out_features;
    int state_len;
    // Output feature size of the linear layer.  Dictated by state_len and whether
    // blank scores are explicitly stored in the linear layer output.
    int outsize;
    float blank_score;
    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    float scale = 1.0f;
    int num_features;
    int sample_rate = -1;
    SignalNormalisationParams signal_norm_params;
    std::filesystem::path model_path;

    // Start position for mean Q-score calculation for
    // short reads.
    int32_t mean_qscore_start_pos = -1;

    SampleType sample_type;

    // convolution layer params
    std::vector<ConvParams> convs;

    // Tx Model Params
    std::optional<tx::Params> tx = std::nullopt;

    // True if this model config describes a LSTM model
    bool is_lstm_model() const { return !is_tx_model(); }
    // True if this model config describes a transormer model
    bool is_tx_model() const { return tx.has_value(); };

    std::string to_string() const;
};

// True if this config at path describes a transformer model
bool is_tx_model_config(const std::filesystem::path& path);

CRFModelConfig load_model_config(const std::filesystem::path& path);

bool is_rna_model(const CRFModelConfig& model_config);
bool is_duplex_model(const CRFModelConfig& model_config);

namespace tx {
CRFModelConfig load_tx_model_config(const std::filesystem::path& path);
}

}  // namespace dorado::basecall