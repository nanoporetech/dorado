#include "decode/ModBaseDecoder.h"

#include "TestUtils.h"
#include "basecall/DecodedChunk.h"
#include "config/ModBaseModelConfig.h"

#include <ATen/ops/arange.h>
#include <ATen/ops/zeros.h>
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#define CUT_TAG "[ModBaseDecoder]"

namespace {

using dorado::basecall::decode::DecodedChunk;
using dorado::basecall::decode::ModBaseDecoder;

constexpr int64_t T = 100;
constexpr int64_t NUM_HEAD_STATES = 2;  // canonical + 6mA

std::vector<dorado::config::ModBaseModelConfig> load_configs() {
    return {dorado::config::load_modbase_model_config(get_data_dir("model_configs") /
                                                      "latent_space_6mA")};
}

// A base is emitted every other timestep.
DecodedChunk make_chunk(const std::string &sequence) {
    DecodedChunk chunk;
    chunk.sequence = sequence;
    chunk.qstring = std::string(sequence.size(), '!');
    chunk.moves.assign(T, 0);
    for (size_t i = 0; i < sequence.size(); ++i) {
        chunk.moves[2 * i] = 1;
    }
    return chunk;
}

// Scores are exactly representable in float16, so the CUDA (half) and CPU (float) model outputs
// carry identical values.
at::Tensor make_scores(const int64_t N) {
    const auto values = at::arange(N * T * NUM_HEAD_STATES, at::kFloat).remainder_(64).div_(64);
    return values.reshape({N, T * NUM_HEAD_STATES}).to(at::kHalf);
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " float and half scores decode identically", CUT_TAG) {
    const auto configs = load_configs();
    const ModBaseDecoder decoder(configs);
    const std::string sequence = "ACGTACGTAACCGGTTAACGTACGTAACCGGTTAACGTACGTAACCGGTT";
    const auto chunk = make_chunk(sequence);
    const auto half_scores = make_scores(1);
    const auto float_scores = half_scores.to(at::kFloat);

    const std::vector<at::Tensor> half_heads{half_scores[0]};
    const std::vector<at::Tensor> float_heads{float_scores[0]};
    const auto [half_probs, half_hits] = decoder.decode(half_heads, chunk.sequence, chunk.moves);
    const auto [float_probs, float_hits] = decoder.decode(float_heads, chunk.sequence, chunk.moves);

    CATCH_CHECK(half_probs == float_probs);
    CATCH_CHECK(half_hits == float_hits);

    // Hits must exist and only fall on the modified base.
    CATCH_REQUIRE(half_hits.size() == sequence.size());
    int num_hits = 0;
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (half_hits[i]) {
            CATCH_CHECK(sequence[i] == 'A');
            ++num_hits;
        }
    }
    CATCH_CHECK(num_hits > 0);
    CATCH_CHECK(half_probs.size() == sequence.size() * (4 + 1));
}

CATCH_TEST_CASE(CUT_TAG " no modbase configs decodes to nothing", CUT_TAG) {
    const ModBaseDecoder decoder({});
    CATCH_CHECK(decoder.empty());
    const auto chunk = make_chunk("ACGT");
    const auto [probs, hits] = decoder.decode({}, chunk.sequence, chunk.moves);
    CATCH_CHECK(probs.empty());
    CATCH_CHECK(hits.empty());
}
