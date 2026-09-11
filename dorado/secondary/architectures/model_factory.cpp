
#include "secondary/architectures/model_factory.h"

#include "model_gru.h"
#include "model_latent_space_lstm.h"
#include "model_slot_attention_consensus.h"
#include "model_variant_perceiver.h"
#include "secondary/architectures/model_config_validation.h"
#include "secondary/architectures/model_weights.h"
#include "secondary/features/encoder_factory.h"
#include "utils/container_utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <unordered_map>
#include <vector>

namespace dorado::secondary {

ModelType parse_model_type(const std::string& type) {
    if (type == "GRUModel") {
        return ModelType::GRU;
    } else if (type == "LatentSpaceLSTM") {
        return ModelType::LATENT_SPACE_LSTM;
    } else if (type == "SlotAttentionConsensus") {
        return ModelType::SLOT_ATTENTION_CONSENSUS;
    } else if (type == "VariantPerceiver") {
        return ModelType::VARIANT_PERCEIVER;
    }
    throw std::runtime_error{"Unknown model type: '" + type + "'!"};
}

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config,
                                              const ParameterLoadingStrategy param_strategy) {
    const auto get_value = [&config](const std::string& key) -> std::string {
        return get_model_config_model_value(config, key);
    };

    const auto get_bool_value = [&config](const std::string& key) -> bool {
        return get_model_config_model_value(config, key) == "true";
    };

    const ModelType model_type = parse_model_type(config.model_type);

    const FeatureColumnMap feature_column_map = feature_column_map_factory(config);

    std::shared_ptr<ModelTorchBase> model;

    if ((config.model_file != "model.pt") && (config.model_file != "weights.pt")) {
        throw std::runtime_error{"Unexpected weights/model file name! model_file = '" +
                                 config.model_file.string() +
                                 "', expected either 'model.pt' or 'weights.pt'."};
    }

    if (model_type == ModelType::GRU) {
        spdlog::debug("Constructing a GRU model.");

        const int32_t num_features = std::stoi(get_value("num_features"));
        const int32_t num_classes = std::stoi(get_value("num_classes"));
        const int32_t gru_size = std::stoi(get_value("gru_size"));
        const int32_t n_layers = std::stoi(get_value("n_layers"));
        const bool bidirectional = get_bool_value("bidirectional");

        model = ModelGRU::make<ModelGRU>(num_features, num_classes, gru_size, n_layers,
                                         bidirectional);

    } else if (model_type == ModelType::LATENT_SPACE_LSTM) {
        spdlog::debug("Constructing a LATENT_SPACE_LSTM model.");

        const int32_t num_classes = std::stoi(get_value("num_classes"));
        const int32_t lstm_size = std::stoi(get_value("lstm_size"));
        const int32_t cnn_size = std::stoi(get_value("cnn_size"));
        const std::string pooler_type = get_value("pooler_type");
        const int32_t bases_alphabet_size = std::stoi(get_value("bases_alphabet_size"));
        const int32_t bases_embedding_size = std::stoi(get_value("bases_embedding_size"));
        const std::vector<int32_t> kernel_sizes =
                utils::parse_int32_vector(get_value("kernel_sizes"), ',');
        const bool use_dwells = get_bool_value("use_dwells");
        const bool bidirectional = get_bool_value("bidirectional");

        model = ModelLatentSpaceLSTM::make<ModelLatentSpaceLSTM>(
                num_classes, lstm_size, cnn_size, kernel_sizes, pooler_type, use_dwells,
                bases_alphabet_size, bases_embedding_size, bidirectional, feature_column_map);

    } else if (model_type == ModelType::SLOT_ATTENTION_CONSENSUS) {
        spdlog::debug("Constructing a SLOT_ATTENTION_CONSENSUS model.");

        const int32_t num_slots = std::stoi(get_value("num_slots"));
        const int32_t classes_per_slot = std::stoi(get_value("classes_per_slot"));
        const int32_t read_embedding_size = std::stoi(get_value("read_embedding_size"));
        const int32_t cnn_size = std::stoi(get_value("cnn_size"));
        const std::vector<int32_t> kernel_sizes =
                utils::parse_int32_vector(get_value("kernel_sizes"), ',');
        const std::string pooler_type = get_value("pooler_type");
        const bool use_mapqc = get_bool_value("use_mapqc");
        const bool use_dwells = get_bool_value("use_dwells");
        const bool use_haplotags = get_bool_value("use_haplotags");
        const bool use_snp_qv = get_bool_value("use_snp_qv");
        const int32_t bases_alphabet_size = std::stoi(get_value("bases_alphabet_size"));
        const int32_t bases_embedding_size = std::stoi(get_value("bases_embedding_size"));
        const bool add_lstm = get_bool_value("add_lstm");
        const bool use_reference = get_bool_value("use_reference");

        const std::unordered_map<std::string, std::string> pooler_args;

        model = ModelSlotAttentionConsensus::make<ModelSlotAttentionConsensus>(
                num_slots, classes_per_slot, read_embedding_size, cnn_size, kernel_sizes,
                pooler_type, pooler_args, use_mapqc, use_dwells, use_haplotags, use_snp_qv,
                bases_alphabet_size, bases_embedding_size, add_lstm, use_reference,
                feature_column_map);

        // The SlotAttentionConsensus model normalizes internally because of phasing, so
        // deactivate normalization after phasing.
        model->set_normalise(false);

    } else if (model_type == ModelType::VARIANT_PERCEIVER) {
        spdlog::debug("Constructing a VARIANT_PERCEIVER model.");

        const int32_t read_max_depth = std::stoi(get_value("read_max_depth"));
        const int32_t ploidy = std::stoi(get_value("ploidy"));
        const int32_t num_classes = std::stoi(get_value("num_classes"));
        const int32_t cnn_size = std::stoi(get_value("cnn_size"));
        const std::vector<int32_t> kernel_sizes =
                utils::parse_int32_vector(get_value("kernel_sizes"), ',');
        const int32_t dimension = std::stoi(get_value("dimension"));
        const int32_t num_blocks = std::stoi(get_value("num_blocks"));
        const int32_t num_heads = std::stoi(get_value("num_heads"));
        const int32_t self_attn_layers_per_block =
                std::stoi(get_value("self_attn_layers_per_block"));

        const bool use_mapqc = get_bool_value("use_mapqc");
        const bool use_dwells = get_bool_value("use_dwells");
        const bool use_haplotags = get_bool_value("use_haplotags");
        const bool use_snp_qv = get_bool_value("use_snp_qv");

        const int32_t bases_alphabet_size = std::stoi(get_value("bases_alphabet_size"));
        const int32_t bases_embedding_size = std::stoi(get_value("bases_embedding_size"));

        const bool use_decoder_lstm = get_bool_value("use_decoder_lstm");
        const bool update_read_embeddings = get_bool_value("update_read_embeddings");
        const bool use_per_read_embedding = get_bool_value("use_per_read_embedding");
        const EmbeddingType embedding_type = parse_embedding_type(get_value("embedding_type"));
        const bool latent_ref_init = parse_latent_init_from_ref(get_value("latent_init_method"));

        model = ModelVariantPerceiver::make<ModelVariantPerceiver>(
                read_max_depth, ploidy, num_classes, cnn_size, kernel_sizes, dimension, num_blocks,
                num_heads, self_attn_layers_per_block, use_mapqc, use_dwells, use_haplotags,
                use_snp_qv, bases_alphabet_size, bases_embedding_size, use_decoder_lstm,
                use_per_read_embedding, embedding_type, update_read_embeddings, latent_ref_init,
                feature_column_map);

    } else {
        throw std::runtime_error("Unsupported model type!");
    }

    // Set the weights of the internally constructed model. This is optional for testing purposes.
    if (param_strategy == ParameterLoadingStrategy::LOAD_WEIGHTS) {
        load_state_dict(*model, config.model_dir / config.model_file,
                        model->get_non_persistent_buffers());
    }

    return model;
}

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config) {
    return model_factory(config, ParameterLoadingStrategy::LOAD_WEIGHTS);
}

}  // namespace dorado::secondary
