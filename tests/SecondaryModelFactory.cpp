#include "secondary/architectures/model_factory.h"

#include <catch2/catch_test_macros.hpp>

// Need to include implementation details to perform dynamic_cast<>s in tests.
#include "../model_gru.h"
#include "../model_latent_space_lstm.h"
#include "../model_slot_attention_consensus.h"

#include <cstdint>
#include <memory>

namespace dorado::secondary::tests {

#define TEST_GROUP "[SecondaryModelFactory]"

CATCH_TEST_CASE("Instantiate models", TEST_GROUP) {
    // Deactivate actual weight loading.
    constexpr ParameterLoadingStrategy PARAM_STRATEGY = ParameterLoadingStrategy::NO_OP;

    CATCH_SECTION("Unknown model, should throw") {
        const ModelConfig config{
                .version = 1,
                .basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                .supported_basecallers = {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0"},
                .model_type = "Unknown",
                .model_file = "wrong_weights_file.pt",
                .model_dir = "",
                .model_kwargs = {},
                .feature_encoder_type = "",
                .feature_encoder_kwargs = {},
                .feature_encoder_dtypes = {},
                .label_scheme_type = "DiploidLabelScheme",
        };

        CATCH_CHECK_THROWS(model_factory(config, PARAM_STRATEGY));
    }

    CATCH_SECTION("Wrong weights file, it should be either weights.pt or model.pt.") {
        const ModelConfig config{
                .version = 1,
                .basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                .supported_basecallers = {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0"},
                .model_type = "GRUModel",
                .model_file = "wrong_weights_file.pt",
                .model_dir = "",
                .model_kwargs =
                        {
                                {"num_features", "10"},
                                {"num_classes", "5"},
                                {"gru_size", "128"},
                                {"n_layers", "2"},
                                {"bidirectional", "true"},
                        },
                .feature_encoder_type = "",
                .feature_encoder_kwargs = {},
                .feature_encoder_dtypes = {},
                .label_scheme_type = "DiploidLabelScheme",
        };

        CATCH_CHECK_THROWS(model_factory(config, PARAM_STRATEGY));
    }

    CATCH_SECTION("TorchScript model cannot be loaded without loading the weights") {
        const ModelConfig config{
                .version = 1,
                .basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                .supported_basecallers = {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0"},
                .model_type = "GRUModel",
                .model_file = "model.pt",
                .model_dir = "",
                .model_kwargs =
                        {
                                {"num_features", "10"},
                                {"num_classes", "5"},
                                {"gru_size", "128"},
                                {"n_layers", "2"},
                                {"bidirectional", "true"},
                        },
                .feature_encoder_type = "",
                .feature_encoder_kwargs = {},
                .feature_encoder_dtypes = {},
                .label_scheme_type = "DiploidLabelScheme",
        };

        CATCH_CHECK_THROWS(model_factory(config, PARAM_STRATEGY));
    }

    CATCH_SECTION("GRU model") {
        const ModelConfig config{
                .version = 1,
                .basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                .supported_basecallers = {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0"},
                .model_type = "GRUModel",
                .model_file = "weights.pt",
                .model_dir = "",
                .model_kwargs =
                        {
                                {"num_features", "10"},
                                {"num_classes", "5"},
                                {"gru_size", "128"},
                                {"n_layers", "2"},
                                {"bidirectional", "true"},
                        },
                .feature_encoder_type = "",
                .feature_encoder_kwargs = {},
                .feature_encoder_dtypes = {},
                .label_scheme_type = "DiploidLabelScheme",
        };

        std::shared_ptr<ModelTorchBase> model = model_factory(config, PARAM_STRATEGY);

        // Positive test.
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelGRU>(model) != nullptr);

        // Negative tests.
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelLatentSpaceLSTM>(model) == nullptr);
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelSlotAttentionConsensus>(model) == nullptr);
    }

    CATCH_SECTION("LSTM model") {
        const ModelConfig config{
                .version = 1,
                .basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                .supported_basecallers = {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0"},
                .model_type = "LatentSpaceLSTM",
                .model_file = "weights.pt",
                .model_dir = "",
                .model_kwargs =
                        {
                                {"num_classes", "5"},
                                {"lstm_size", "384"},
                                {"cnn_size", "64"},
                                {"kernel_sizes", "[ 1, 17,]"},
                                {"pooler_type", "mean"},
                                {"use_dwells", "true"},
                                {"bases_alphabet_size", "6"},
                                {"bases_embedding_size", "6"},
                                {"bidirectional", "false"},
                        },
                .feature_encoder_type = "",
                .feature_encoder_kwargs = {},
                .feature_encoder_dtypes = {},
                .label_scheme_type = "DiploidLabelScheme",
        };

        std::shared_ptr<ModelTorchBase> model = model_factory(config, PARAM_STRATEGY);

        // Positive test.
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelLatentSpaceLSTM>(model) != nullptr);

        // Negative tests.
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelGRU>(model) == nullptr);
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelSlotAttentionConsensus>(model) == nullptr);
    }

    CATCH_SECTION("SlotAttentionConsensus model") {
        const ModelConfig config{
                .version = 1,
                .basecaller_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                .supported_basecallers = {"dna_r10.4.1_e8.2_400bps_hac@v5.0.0"},
                .model_type = "SlotAttentionConsensus",
                .model_file = "weights.pt",
                .model_dir = "",
                .model_kwargs =
                        {
                                {"num_slots", "2"},
                                {"classes_per_slot", "5"},
                                {"read_embedding_size", "192"},
                                {"cnn_size", "64"},
                                {"kernel_sizes", "[ 1, 17,]"},
                                {"pooler_type", "mean"},
                                {"use_mapqc", "true"},
                                {"use_dwells", "true"},
                                {"use_haplotags", "true"},
                                {"bases_alphabet_size", "6"},
                                {"bases_embedding_size", "6"},
                                {"add_lstm", "true"},
                                {"use_reference", "false"},
                        },
                .feature_encoder_type = "",
                .feature_encoder_kwargs = {},
                .feature_encoder_dtypes = {},
                .label_scheme_type = "DiploidLabelScheme",
        };

        std::shared_ptr<ModelTorchBase> model = model_factory(config, PARAM_STRATEGY);

        // Positive test.
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelSlotAttentionConsensus>(model) != nullptr);

        // Negative tests.
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelGRU>(model) == nullptr);
        CATCH_REQUIRE(std::dynamic_pointer_cast<ModelLatentSpaceLSTM>(model) == nullptr);
    }
}

}  // namespace dorado::secondary::tests