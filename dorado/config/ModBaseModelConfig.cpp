#include "config/ModBaseModelConfig.h"

#include "config/common.h"
#include "hts_utils/bam_utils.h"
#include "utils/dev_utils.h"
#include "utils/overloaded.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace keys {
namespace {
// Workaround GCC-13 dangling reference warnings by passing an lvalue instead of a temporary
const std::string MODBASES{"modbases"};
const std::string MOD_BASES{"mod_bases"};
}  // namespace
}  // namespace keys

namespace {
const std::string ERR_STR = "Invalid modbase model parameter in ";

// Indicates that a value has no default and is therefore required
constexpr std::optional<int> REQUIRED = std::nullopt;

// Get an integer value from a toml::value asserting that it is within a closed interval.
// If no default is given then the key must exist in the toml::value.
int get_int_in_range(const toml::value& p,
                     const std::string& key,
                     int min_val,
                     int max_val,
                     std::optional<int> default_val) {
    const int val = default_val.has_value()
                            ? toml::find_or<int>(p, key, std::forward<int>(default_val.value()))
                            : toml::find<int>(p, key);
    if (val < min_val || val > max_val) {
        auto v = std::to_string(val);
        auto r = std::to_string(min_val) + " <= x <= " + std::to_string(max_val);
        throw std::runtime_error("Invalid modbase model value for '" + key + "' found: '" + v +
                                 "' which is not in range [" + r + "]");
    }
    return val;
}

}  // namespace
namespace dorado::config {

namespace modbase {
namespace {
// Normalise `v` by `stride` strictly increasing the value if needed.
int64_t normalise_to_stride(const int64_t v, const int64_t stride) {
    const int64_t remainder = v % stride;
    if (remainder == 0) {
        return v;
    }
    return v + stride - remainder;
}

int stride_product(const std::vector<ConvParams>& cs) {
    return std::accumulate(cs.cbegin(), cs.cend(), 1,
                           [](const int s, const auto& c) { return s * c.stride; });
}

LinearParams parse_linear(const toml::value& segment) {
    LinearParams p;
    p.in_size = toml::find<int>(segment, "in_features");
    p.out_size = toml::find<int>(segment, "out_features");
    return p;
}

LinearUpsampleParams parse_linear_upsample(const toml::value& segment) {
    LinearUpsampleParams p;
    p.size = toml::find<int>(segment, "size");
    p.scale_factor = toml::find<int>(segment, "scale_factor");
    return p;
}

LSTMParams parse_lstm(const toml::value& segment) {
    LSTMParams p;
    p.size = toml::find<int>(segment, "size");
    p.reverse = bool(toml::find<int>(segment, "reverse"));
    return p;
}

std::vector<LSTMParams> parse_lstms(const std::vector<toml::value>& sublayers) {
    std::vector<LSTMParams> lstms;
    for (const auto& sublayer : sublayers) {
        if (sublayer_type(sublayer) == SublayerType::LSTM) {
            lstms.push_back(parse_lstm(sublayer));
        }
    }

    if (lstms.empty()) {
        throw std::runtime_error("Modbase model config has no lstm layers");
    }
    if (lstms.front().reverse) {
        throw std::runtime_error("Modbase model config first lstm layer must be forward");
    }
    for (size_t i = 0; i < (lstms.size() - 1); ++i) {
        if (lstms[i].size != lstms[i + 1].size) {
            throw std::runtime_error("Modbase model config lstm layers unequal sizes");
        }
        if (lstms[i].reverse == lstms[i + 1].reverse) {
            throw std::runtime_error("Modbase model config lstm layers must alternate direction");
        }
    }
    return lstms;
}

ConvParams parse_merge_conv(const std::vector<toml::value>& sublayers) {
    if (sublayers.empty()) {
        throw std::runtime_error("Modbase model config missing enoder sublayers");
    }
    const auto& front = sublayers.front();
    if (sublayer_type(front) != SublayerType::CONVOLUTION) {
        throw std::runtime_error("Modbase model config missing enconder merge convolution");
    }
    return parse_conv_params(front, false);
}

ModulesParams parse_modules_params(const toml::value& config) {
    ModulesParams m;
    m.sequence_convs = parse_convs(toml::find(config, "sequence_encoder", "sublayers").as_array());
    m.signal_convs = parse_convs(toml::find(config, "signal_encoder", "sublayers").as_array());

    const auto& layers = toml::find(config, "encoder", "sublayers").as_array();
    m.merge_conv = parse_merge_conv(layers);
    m.lstms = parse_lstms(layers);

    for (const auto& layer : layers) {
        if (sublayer_type(layer) == SublayerType::LINEAR) {
            m.linear = parse_linear(layer);
        }
        if (sublayer_type(layer) == SublayerType::UPSAMPLE) {
            m.upsample = parse_linear_upsample(layer);
        }
    }

    if (m.lstms.back().size != m.linear.in_size) {
        throw std::runtime_error("Modbase model config lstm and linear size mismatch");
    }

    return m;
}

ModificationParams parse_modification_params(const toml::value& config_toml) {
    const auto& params = toml::find(config_toml, keys::MODBASES);

    std::vector<std::string> codes;
    const auto& mod_bases = toml::find(params, keys::MOD_BASES);
    if (mod_bases.is_string()) {
        // style: mod_bases = "hm" - does not accept chebi codes
        auto mod_base_string = mod_bases.as_string();
        for (const auto& mod_base : mod_base_string) {
            codes.push_back(std::string(1, mod_base));
        }
    } else {
        // style: mod_bases = [ "h", "m",]
        auto mod_base_array = mod_bases.as_array();
        for (const auto& mod_base : mod_base_array) {
            assert(mod_base.is_string());
            codes.push_back(mod_base.as_string());
        }
    }

    std::vector<std::string> long_names;
    long_names.reserve(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) {
        long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }

    const auto motif = toml::find<std::string>(params, "motif");
    const auto motif_offset = static_cast<size_t>(
            get_int_in_range(params, "motif_offset", 0, int(motif.size()), REQUIRED));

    return ModificationParams{std::move(codes), std::move(long_names), motif, motif_offset};
}

ContextParams parse_context_params(const toml::value& config_toml) {
    const auto& params = toml::find(config_toml, keys::MODBASES);

    const int context_before = get_int_in_range(params, "chunk_context_0", 0, 4096, REQUIRED);
    const int context_after = get_int_in_range(params, "chunk_context_1", 1, 4096, REQUIRED);

    const auto bases_before = get_int_in_range(params, "kmer_context_bases_0", 0, 9, REQUIRED);
    const auto bases_after = get_int_in_range(params, "kmer_context_bases_1", 0, 9, REQUIRED);

    const bool reverse = toml::find_or<bool>(params, "reverse_signal", false);
    const bool base_start_justify = toml::find_or<bool>(params, "base_start_justify", false);

    return ContextParams(context_before, context_after, bases_before, bases_after, reverse,
                         base_start_justify);
}

int64_t parse_chunk_size(const toml::value& config_toml, const ContextParams& context) {
    const auto& params = toml::find(config_toml, keys::MODBASES);
    constexpr int MAX_CHUNK_SIZE = 102400;
    const auto chunk_size = utils::get_dev_opt(
            "modbase_chunksize",
            get_int_in_range(params, "chunk_size", static_cast<int>(context.samples),
                             MAX_CHUNK_SIZE, static_cast<int>(context.samples)));
    if (chunk_size < context.samples) {
        throw std::runtime_error(ERR_STR + "context params: 'chunk size < context size'.");
    }
    return chunk_size;
}

RefinementParams parse_refinement_params(const toml::value& config_toml) {
    if (!config_toml.contains("refinement")) {
        return RefinementParams{};
    }

    const auto segment = toml::find(config_toml, "refinement");

    bool do_rough_rescale = toml::find<int>(segment, "refine_do_rough_rescale") == 1;
    if (!do_rough_rescale) {
        return RefinementParams{};
    }

    const int center_index = get_int_in_range(segment, "refine_kmer_center_idx", 0, 19, REQUIRED);
    return RefinementParams(center_index);
}

CommonModelParams parse_common_model_params(const toml::value& config_toml) {
    const auto params = toml::find(config_toml, "model_params");
    constexpr int max_size = 4096, max_kmer = 19, max_features = 10, max_stride = 6;
    const auto size = get_int_in_range(params, "size", 1, max_size, REQUIRED);
    const auto kmer_len = get_int_in_range(params, "kmer_len", 1, max_kmer, REQUIRED);
    const auto num_out = get_int_in_range(params, "num_out", 1, max_features, REQUIRED);
    const auto stride = get_int_in_range(params, "stride", 1, max_stride, 3);
    return CommonModelParams{size, kmer_len, num_out, stride};
}

ConvOnlyModelConfig parse_conv_v1(const std::filesystem::path& model_path,
                                  const toml::value& config_toml) {
    auto common = parse_common_model_params(config_toml);
    auto mods = parse_modification_params(config_toml);
    auto context = parse_context_params(config_toml);
    auto refine = parse_refinement_params(config_toml);
    return ConvOnlyModelConfig{
            model_path, std::move(common), std::move(mods), std::move(context), std::move(refine),
    };
}

ConvLSTMModelConfig parse_lstm_v1(const std::filesystem::path& model_path,
                                  const toml::value& config_toml) {
    auto common = parse_common_model_params(config_toml);
    auto mods = parse_modification_params(config_toml);
    auto context = parse_context_params(config_toml);
    auto refine = parse_refinement_params(config_toml);
    return ConvLSTMModelConfig{
            model_path, std::move(common), std::move(mods), std::move(context), std::move(refine),
    };
}

ConvLSTMV2ModelConfig parse_lstm_v2(const std::filesystem::path& model_path,
                                    const toml::value& config_toml) {
    auto common = parse_common_model_params(config_toml);
    auto mods = parse_modification_params(config_toml);
    auto context = parse_context_params(config_toml);
    auto refine = parse_refinement_params(config_toml);
    const auto raw_chunk_size = parse_chunk_size(config_toml, context);
    const auto chunk_size = normalise_to_stride(raw_chunk_size, common.stride);
    spdlog::trace("Normalised modbase chunk size for stride: {} - {} -> {}", common.stride,
                  raw_chunk_size, chunk_size);
    return ConvLSTMV2ModelConfig{
            model_path,      std::move(common),  ChunkedModelParams{chunk_size},
            std::move(mods), std::move(context), std::move(refine),
    };
}

ConvLSTMV3ModelConfig parse_lstm_v3(const std::filesystem::path& model_path,
                                    const toml::value& config_toml) {
    auto common = parse_common_model_params(config_toml);
    const auto params = toml::find(config_toml, "model_params");
    constexpr int max_stride = 6;
    const auto sequence_stride =
            get_int_in_range(params, "sequence_stride", 1, max_stride, common.stride);
    auto modules = parse_modules_params(config_toml);

    auto mods = parse_modification_params(config_toml);
    auto context = parse_context_params(config_toml);
    auto refine = parse_refinement_params(config_toml);
    const auto chunk_size = parse_chunk_size(config_toml, context);
    return ConvLSTMV3ModelConfig{
            model_path,         std::move(common), ChunkedModelParams{chunk_size},
            std::move(modules), sequence_stride,   std::move(mods),
            std::move(context), std::move(refine),
    };
}

ModBaseModelConfig parse_model_config(const std::filesystem::path& model_path,
                                      const toml::value& config_toml) {
    const ModelType type =
            model_type_from_string(toml::find<std::string>(config_toml, "general", "model"));
    switch (type) {
    case ModelType::CONV_V1:
        return parse_conv_v1(model_path, config_toml);
    case ModelType::CONV_LSTM_V1:
        return parse_lstm_v1(model_path, config_toml);
    case ModelType::CONV_LSTM_V2:
        return parse_lstm_v2(model_path, config_toml);
    case ModelType::CONV_LSTM_V3:
        return parse_lstm_v3(model_path, config_toml);
    case ModelType::UNKNOWN:
        throw std::runtime_error(ERR_STR + "general params: 'model type is unknown'");
    }
    throw std::runtime_error(ERR_STR + "general params: 'model type is unknown'");
}
}  // namespace

std::string to_string(const ModelType& model_type) {
    switch (model_type) {
    case ModelType::CONV_LSTM_V1:
        return std::string("conv_lstm");
    case ModelType::CONV_LSTM_V2:
        return std::string("conv_lstm_v2");
    case ModelType::CONV_LSTM_V3:
        return std::string("conv_lstm_v3");
    case ModelType::CONV_V1:
        return std::string("conv_v1");
    case ModelType::UNKNOWN:
        return std::string("UNKNOWN");
    }
    throw std::runtime_error("Invalid modbase model type");
};

ModelType model_type_from_string(const std::string& model_type) {
    if (model_type == "conv_lstm") {
        return ModelType::CONV_LSTM_V1;
    }
    if (model_type == "conv_lstm_v2") {
        return ModelType::CONV_LSTM_V2;
    }
    if (model_type == "conv_lstm_v3") {
        return ModelType::CONV_LSTM_V3;
    }
    if (model_type == "conv_only" || model_type == "conv_v1") {
        return ModelType::CONV_V1;
    }
    return ModelType::UNKNOWN;
}

ModelType get_modbase_model_type(const std::filesystem::path& path) {
    try {
        const auto config_toml = toml::parse(path / "config.toml");
        if (!config_toml.contains("general")) {
            return ModelType::UNKNOWN;
        }
        return model_type_from_string(toml::find<std::string>(config_toml, "general", "model"));
    } catch (std::exception&) {
        return ModelType::UNKNOWN;
    }
}

int ModulesParams::sequence_stride() const { return stride_product(sequence_convs); };
int ModulesParams::signal_stride() const { return stride_product(signal_convs); };
int ModulesParams::stride_ratio() const {
    const auto seq = sequence_stride();
    const auto sig = signal_stride();

    if (sig < seq) {
        throw std::runtime_error(
                "modbase sequence stride must be less than or equal to signal stride");
    }
    if (sig % seq != 0) {
        throw std::runtime_error(
                "modbase signal stride must be evenly divisible by sequence stride");
    }
    return sig / seq;
};

CommonModelParams::CommonModelParams(int size_, int kmer_len_, int num_out_, int stride_)
        : size(size_), kmer_len(kmer_len_), num_out(num_out_), stride(stride_) {
    if (size < 1 || kmer_len < 1 || num_out < 1 || stride < 1) {
        throw std::runtime_error(ERR_STR + "general params: 'negative or zero value'.");
    }
    if (kmer_len % 2 != 1) {
        throw std::runtime_error(ERR_STR + "general params: 'kmer_length is not odd'.");
    }
}

ChunkedModelParams::ChunkedModelParams(int64_t chunk_size_) : chunk_size(chunk_size_) {
    if (chunk_size < 1) {
        throw std::runtime_error(ERR_STR + "chunked model params: 'negative or zero chunk size'.");
    }
}

ModificationParams::ModificationParams(std::vector<std::string> codes_,
                                       std::vector<std::string> long_names_,
                                       std::string motif_,
                                       const size_t motif_offset_)
        : codes(std::move(codes_)),
          long_names(std::move(long_names_)),
          count(codes.size()),
          motif(std::move(motif_)),
          motif_offset(motif_offset_),
          base(get_canonical_base_name(motif, motif_offset)),
          base_id(utils::BaseInfo::BASE_IDS[base]) {
    if (codes.empty()) {
        throw std::runtime_error(ERR_STR + "mods params: 'empty modifications.");
    }
    if (long_names.empty()) {
        throw std::runtime_error(ERR_STR + "mods params: 'empty long names.");
    }
    if (codes.size() != long_names.size()) {
        throw std::runtime_error(ERR_STR + "mods params: 'mods and names size mismatch.");
    }

    for (const auto& code : codes) {
        if (!utils::validate_bam_tag_code(code)) {
            std::string e = ERR_STR + "mods params: 'invalid mod code ";
            throw std::runtime_error(e + code + "'.");
        }
    }
}

char ModificationParams::get_canonical_base_name(const std::string& motif, size_t motif_offset) {
    if (motif.size() < motif_offset) {
        throw std::runtime_error(ERR_STR + "mods params: 'invalid motif offset'.");
    }

    // Assert a canonical base is at motif[motif_offset]
    constexpr std::string_view canonical_bases = "ACGT";
    std::string motif_base = motif.substr(motif_offset, 1);
    if (canonical_bases.find(motif_base) == std::string::npos) {
        throw std::runtime_error(ERR_STR + "mods params: 'invalid motif base " + motif_base + "'.");
    }

    return motif_base[0];
}

ContextParams::ContextParams(int64_t samples_before_,
                             int64_t samples_after_,
                             int bases_before_,
                             int bases_after_,
                             bool reverse_,
                             bool base_start_justify_)
        : samples_before(samples_before_),
          samples_after(samples_after_),
          samples(samples_before + samples_after),
          bases_before(bases_before_),
          bases_after(bases_after_),
          kmer_len(bases_before_ + bases_after_ + 1),
          reverse(reverse_),
          base_start_justify(base_start_justify_) {
    if (samples_before < 0 || samples_after < 0) {
        throw std::runtime_error(ERR_STR + "context params: 'negative context samples'.");
    }
    if (bases_before < 1 || bases_after < 1) {
        throw std::runtime_error(ERR_STR + "context params: 'negative or zero context bases'.");
    }
}

ContextParams ContextParams::normalised(const int stride) const {
    const int64_t sb = normalise_to_stride(samples_before, stride);
    const int64_t sa = normalise_to_stride(samples_after, stride);
    spdlog::trace(
            "Normalised modbase context for stride: {} - [{}, {}, {}] -> [{}, {}, {}] @ "
            "[samples_before, samples_after, samples]",
            stride, samples_before, samples_after, samples, sb, sa, sb + sa);
    return ContextParams(sb, sa, bases_before, bases_after, reverse, base_start_justify);
}

RefinementParams::RefinementParams(int center_idx_)
        : do_rough_rescale(true), center_idx(static_cast<size_t>(center_idx_)) {
    if (center_idx_ < 0) {
        throw std::runtime_error(ERR_STR + "refinement params: 'negative center index'.");
    }
}

ModBaseModelConfigData::ModBaseModelConfigData(std::filesystem::path model_path_,
                                               ModificationParams mods_,
                                               ContextParams context_,
                                               RefinementParams refine_,
                                               int kmer_len_)
        : model_path(std::move(model_path_)),
          mods(std::move(mods_)),
          context(std::move(context_)),
          refine(std::move(refine_)) {
    // Kmer length is duplicated in modbase model configs - check they match
    if (kmer_len_ != context.kmer_len) {
        auto kl_a = std::to_string(kmer_len_);
        auto kl_b = std::to_string(context.kmer_len);
        throw std::runtime_error(ERR_STR + "config: 'inconsistent kmer_len: " + kl_a +
                                 " != " + kl_b + "'.");
    }
}

ConvOnlyModelConfig::ConvOnlyModelConfig(std::filesystem::path model_path_,
                                         CommonModelParams general_,
                                         ModificationParams mods_,
                                         ContextParams context_,
                                         RefinementParams refine_)
        : data(std::move(model_path_),
               std::move(mods_),
               std::move(context_),
               std::move(refine_),
               general_.kmer_len),
          general(std::move(general_)) {}

ConvLSTMModelConfig::ConvLSTMModelConfig(std::filesystem::path model_path_,
                                         CommonModelParams general_,
                                         ModificationParams mods_,
                                         ContextParams context_,
                                         RefinementParams refine_)
        : data(std::move(model_path_),
               std::move(mods_),
               std::move(context_),
               std::move(refine_),
               general_.kmer_len),
          general(std::move(general_)) {}

ConvLSTMV2ModelConfig::ConvLSTMV2ModelConfig(std::filesystem::path model_path_,
                                             CommonModelParams general_,
                                             ChunkedModelParams chunked_,
                                             ModificationParams mods_,
                                             ContextParams context_,
                                             RefinementParams refine_)
        : data(std::move(model_path_),
               std::move(mods_),
               context_.normalised(general_.stride),
               std::move(refine_),
               general_.kmer_len),
          general(std::move(general_)),
          chunked(std::move(chunked_)) {
    if (chunked.chunk_size < data.context.samples) {
        throw std::runtime_error(ERR_STR + "context params: 'chunk size < context size'.");
    }
}

ConvLSTMV3ModelConfig::ConvLSTMV3ModelConfig(std::filesystem::path model_path_,
                                             CommonModelParams general_,
                                             ChunkedModelParams chunked_,
                                             ModulesParams modules_,
                                             int sequence_stride_,
                                             ModificationParams mods_,
                                             ContextParams context_,
                                             RefinementParams refine_)
        : data(std::move(model_path_),
               std::move(mods_),
               std::move(context_),
               std::move(refine_),
               general_.kmer_len),
          general(std::move(general_)),
          chunked(std::move(chunked_)),
          modules(std::move(modules_)),
          sequence_stride(sequence_stride_) {
    if (sequence_stride < 1) {
        throw std::runtime_error(ERR_STR + "v3 model params: 'negative or zero sequence stride'.");
    }
    if (general.stride % sequence_stride != 0) {
        throw std::runtime_error(
                "Modbase model config signal stride must be evenly divisible by sequence stride");
    }
    if ((general.size != modules.lstms.front().size) ||
        (modules.lstms.front().size != modules.lstms.back().size)) {
        throw std::runtime_error("Modbase model config lstm size mismatch");
    }
    if (general.stride != stride_product(modules.signal_convs)) {
        throw std::runtime_error("Modbase model config signal convolution stride mismatch");
    }
    if (sequence_stride != stride_product(modules.sequence_convs)) {
        throw std::runtime_error("Modbase model config sequence convolution stride mismatch");
    }
    if (general.num_out != modules.linear.out_size) {
        throw std::runtime_error("Modbase model config linear and num_out mismatch");
    }
}

}  // namespace modbase

const modbase::ModBaseModelConfigData& get_model_config_data(const ModBaseModelConfig& config) {
    return std::visit(
            [](const auto& model) -> const modbase::ModBaseModelConfigData& { return model.data; },
            config);
}

const modbase::CommonModelParams& get_common_model_params(const ModBaseModelConfig& config) {
    return std::visit(
            [](const auto& model) -> const modbase::CommonModelParams& { return model.general; },
            config);
}

modbase::ModelType get_model_type(const ModBaseModelConfig& config) {
    return std::visit(
            utils::overloaded{
                    [](const modbase::ConvOnlyModelConfig&) { return modbase::ModelType::CONV_V1; },
                    [](const modbase::ConvLSTMModelConfig&) {
                        return modbase::ModelType::CONV_LSTM_V1;
                    },
                    [](const modbase::ConvLSTMV2ModelConfig&) {
                        return modbase::ModelType::CONV_LSTM_V2;
                    },
                    [](const modbase::ConvLSTMV3ModelConfig&) {
                        return modbase::ModelType::CONV_LSTM_V3;
                    },
            },
            config);
}

bool is_chunked_input_model(const ModBaseModelConfig& config) {
    return std::holds_alternative<modbase::ConvLSTMV2ModelConfig>(config) ||
           std::holds_alternative<modbase::ConvLSTMV3ModelConfig>(config);
}

const modbase::ChunkedModelParams& get_chunked_model_params(const ModBaseModelConfig& config) {
    return std::visit(
            utils::overloaded{
                    [](const modbase::ConvLSTMV2ModelConfig& model)
                            -> const modbase::ChunkedModelParams& { return model.chunked; },
                    [](const modbase::ConvLSTMV3ModelConfig& model)
                            -> const modbase::ChunkedModelParams& { return model.chunked; },
                    [](const auto&) -> const modbase::ChunkedModelParams& {
                        throw std::runtime_error("Modbase model does not accept chunked input");
                    },
            },
            config);
}

int get_chunked_model_stride_ratio(const ModBaseModelConfig& config) {
    return std::visit(
            utils::overloaded{
                    [](const modbase::ConvLSTMV2ModelConfig&) -> int { return 1; },
                    [](const modbase::ConvLSTMV3ModelConfig& model) -> int {
                        return model.general.stride / model.sequence_stride;
                    },
                    [](const auto&) -> int {
                        throw std::runtime_error("Modbase model does not accept chunked input");
                    },
            },
            config);
}

std::tuple<int64_t, int64_t> get_chunked_sequence_input_TC(
        const modbase::ConvLSTMV3ModelConfig& config) {
    return {config.chunked.chunk_size / (config.general.stride / config.sequence_stride),
            config.data.context.kmer_len * 4};
}

std::tuple<int64_t, int64_t> get_chunked_signal_input_TC(
        const modbase::ConvLSTMV3ModelConfig& config) {
    return {config.chunked.chunk_size, 1};
}

std::tuple<int64_t, int64_t> get_chunked_sequence_input_TC(const ModBaseModelConfig& config) {
    return std::visit(
            utils::overloaded{
                    [](const modbase::ConvLSTMV2ModelConfig& model) {
                        return std::tuple{model.chunked.chunk_size,
                                          int64_t(model.data.context.kmer_len * 4)};
                    },
                    [](const modbase::ConvLSTMV3ModelConfig& model) {
                        return get_chunked_sequence_input_TC(model);
                    },
                    [](const auto&) -> std::tuple<int64_t, int64_t> {
                        throw std::runtime_error("Modbase model does not accept chunked input");
                    },
            },
            config);
}

std::tuple<int64_t, int64_t> get_chunked_signal_input_TC(const ModBaseModelConfig& config) {
    return std::visit(
            utils::overloaded{
                    [](const modbase::ConvLSTMV2ModelConfig& model) {
                        return std::tuple{model.chunked.chunk_size, int64_t(1)};
                    },
                    [](const modbase::ConvLSTMV3ModelConfig& model) {
                        return get_chunked_signal_input_TC(model);
                    },
                    [](const auto&) -> std::tuple<int64_t, int64_t> {
                        throw std::runtime_error("Modbase model does not accept chunked input");
                    },
            },
            config);
}

std::tuple<int64_t, int64_t> get_chunked_output_TC(const ModBaseModelConfig& config) {
    const auto& params = get_chunked_model_params(config);
    const auto& common = get_common_model_params(config);
    return {params.chunk_size / common.stride, common.num_out};
}

const modbase::ModBaseModelConfigData& get_context_model_config(const ModBaseModelConfig& config) {
    assert(!is_chunked_input_model(config));
    if (const auto* conv_only = std::get_if<modbase::ConvOnlyModelConfig>(&config)) {
        return conv_only->data;
    }
    return std::get<modbase::ConvLSTMModelConfig>(config).data;
}

bool is_modbase_model(const std::filesystem::path& path) {
    return modbase::get_modbase_model_type(path) != modbase::ModelType::UNKNOWN;
}

ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path) {
    const auto config_toml = toml::parse(model_path / "config.toml");
    return modbase::parse_model_config(model_path, config_toml);
}

ModBaseInfo get_modbase_info(
        const std::vector<std::reference_wrapper<const ModBaseModelConfig>>& base_mod_params) {
    struct ModelInfo {
        std::vector<std::string> long_names;
        std::vector<std::string> alphabet;
        std::string motif;
        int motif_offset;
        size_t base_counts = 1;
    };

    const std::string allowed_bases = "ACGT";
    std::array<ModelInfo, utils::BaseInfo::NUM_BASES> model_info;
    for (int b = 0; b < utils::BaseInfo::NUM_BASES; ++b) {
        model_info[b].alphabet.emplace_back(1, allowed_bases[b]);
    }

    for (const auto& params_ref : base_mod_params) {
        const auto& params = get_model_config_data(params_ref.get()).mods;
        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            throw std::runtime_error("Invalid base in modbase model metadata.");
        }
        auto& map_entry = model_info[utils::BaseInfo::BASE_IDS[base]];
        map_entry.long_names = params.long_names;
        map_entry.alphabet.insert(map_entry.alphabet.end(), params.codes.begin(),
                                  params.codes.end());
        map_entry.base_counts = params.count + 1;
    }

    ModBaseInfo result;
    size_t index = 0;
    for (const auto& info : model_info) {
        for (const auto& name : info.long_names) {
            if (!result.long_names.empty()) {
                result.long_names += ' ';
            }
            result.long_names += name;
        }
        result.alphabet.insert(result.alphabet.end(), info.alphabet.begin(), info.alphabet.end());
        result.base_counts[index++] = info.base_counts;
    }

    return result;
}

void check_modbase_multi_model_compatibility(
        const std::vector<std::filesystem::path>& modbase_models) {
    if (modbase_models.size() < 2) {
        return;
    }

    std::string err_msg = "";
    for (size_t i = 0; i < modbase_models.size(); i++) {
        const auto ref_model = load_modbase_model_config(modbase_models[i]);
        const auto& ref_params = get_model_config_data(ref_model).mods;
        const auto& ref_motif = ref_params.motif[ref_params.motif_offset];

        for (size_t j = i + 1; j < modbase_models.size(); j++) {
            const auto query_model = load_modbase_model_config(modbase_models[j]);
            const auto& query_params = get_model_config_data(query_model).mods;
            const auto& query_motif = query_params.motif[query_params.motif_offset];

            if (ref_motif == query_motif) {
                err_msg += modbase_models[i].filename().string() + " and " +
                           modbase_models[j].filename().string() +
                           " have overlapping motifs on the canonical base '" + ref_motif +
                           "' and only one modbase model motif per canonical base is allowed.";
            }
        }
    }

    if (!err_msg.empty()) {
        throw std::runtime_error("Incompatible modbase models detected: " + err_msg);
    }
}

}  // namespace dorado::config
