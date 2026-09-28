#include "../dorado/modbase/nn/ModBaseLatentSpaceV1SequenceModel.h"
#include "TestUtils.h"
#include "config/ModBaseModelConfig.h"
#include "modbase/encode_kmer.h"
#include "modbase/nn/ModBaseModel.h"
#include "utils/sequence_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>  // IWYU: keep for CATCH_CHECK_THROWS_WITH macro

#include <cstdint>
#include <tuple>
#include <vector>

#define TEST_GROUP "[modbase_latent_space_v1_model]"

namespace {

static_assert(std::tuple_size_v<dorado::modbase::ModBaseLatentSpaceModels> == 3);

constexpr int64_t COMPONENTS = 3;
constexpr int64_t BATCH_SIZE = 2;
constexpr int64_t TIME_STEPS = 8;

size_t decoded_offset(int64_t component, int64_t batch, int64_t timestep) {
    return static_cast<size_t>(((component * BATCH_SIZE) + batch) * TIME_STEPS + timestep);
}

void set_decoder_row(std::vector<int8_t>& decoded,
                     int64_t component,
                     int64_t batch,
                     const std::vector<int8_t>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        decoded.at(decoded_offset(component, batch, static_cast<int64_t>(i))) = values.at(i);
    }
}

}  // namespace

CATCH_TEST_CASE("Latent-space sequence model matches the modbase k-mer encoder", TEST_GROUP) {
    const auto config_path = get_data_dir("model_configs") / "latent_space_6mA";
    const auto config = dorado::config::load_modbase_model_config(config_path);
    const auto& latent_config = std::get<dorado::config::modbase::LatentSpaceV1ModelConfig>(config);
    dorado::modbase::model::ModBaseLatentSpaceV1SequenceModel model(latent_config);

    std::vector<int8_t> decoded(COMPONENTS * BATCH_SIZE * TIME_STEPS, 0);
    // Batch 0: A occupies [0, 2), C [2, 5), and G [5, 8).
    set_decoder_row(decoded, 0, 0, {1, 0, 1, 0, 0, 1, 0, 0});
    set_decoder_row(decoded, 1, 0, {'A', 'C', 'G'});
    // Batch 1: T occupies [0, 2), then A [2, 8).
    set_decoder_row(decoded, 0, 1, {1, 0, 1, 0, 0, 0, 0, 0});
    set_decoder_row(decoded, 1, 1, {'T', 'A'});
    // qstring is deliberately ignored by the sequence model.
    set_decoder_row(decoded, 2, 0, {'!', '"', '#'});
    set_decoder_row(decoded, 2, 1, {'$', '%'});

    const auto options = torch::TensorOptions().dtype(torch::kInt8);
    const auto input =
            torch::from_blob(decoded.data(), {COMPONENTS, BATCH_SIZE, TIME_STEPS}, options).clone();
    const auto result = model->forward(input);

    std::vector<int8_t> expected;
    for (const auto& [sequence, mapping] :
         std::vector<std::pair<std::string, std::vector<uint64_t>>>{{"ACG", {0, 2, 5, 8}},
                                                                    {"TA", {0, 2, 8}}}) {
        const auto encoded = dorado::modbase::encode_kmer_chunk(
                dorado::utils::sequence_to_ints(sequence), mapping, {},
                static_cast<size_t>(latent_config.general.kmer_len), TIME_STEPS, 0, true);
        expected.insert(expected.end(), encoded.begin(), encoded.end());
    }
    const auto expected_tensor = torch::from_blob(expected.data(), result.sizes(), options).clone();

    CATCH_CHECK(result.scalar_type() == torch::kInt8);
    CATCH_CHECK(result.sizes() ==
                torch::IntArrayRef({BATCH_SIZE, TIME_STEPS, latent_config.general.kmer_len * 4}));
    CATCH_CHECK(result.equal(expected_tensor));
}

CATCH_TEST_CASE("Latent-space sequence model validates decoder input", TEST_GROUP) {
    const auto config_path = get_data_dir("model_configs") / "latent_space_6mA";
    const auto config = dorado::config::load_modbase_model_config(config_path);
    const auto& latent_config = std::get<dorado::config::modbase::LatentSpaceV1ModelConfig>(config);
    dorado::modbase::model::ModBaseLatentSpaceV1SequenceModel model(latent_config);

    CATCH_CHECK_THROWS_WITH(model->forward(torch::zeros({BATCH_SIZE, TIME_STEPS}, torch::kInt8)),
                            "ModBaseLatentSpaceV1SequenceModel input must have shape [3, N, T]");
    CATCH_CHECK_THROWS_WITH(
            model->forward(torch::zeros({COMPONENTS, BATCH_SIZE, TIME_STEPS}, torch::kFloat32)),
            "ModBaseLatentSpaceV1SequenceModel input must have int8 dtype");
}
