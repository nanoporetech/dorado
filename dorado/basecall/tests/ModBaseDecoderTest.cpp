#include "decode/ModBaseDecoder.h"

#include "TestUtils.h"
#include "config/ModBaseModelConfig.h"

#include <ATen/ops/arange.h>
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#define CUT_TAG "[ModBaseDecoder]"

namespace {

using dorado::basecall::decode::DecodedChunk;
using dorado::basecall::decode::ModBaseDecoder;
using dorado::basecall::decode::pack_decoded_chunks;

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

CATCH_TEST_CASE(CUT_TAG " batch decode matches per-chunk decode", CUT_TAG) {
    const auto configs = load_configs();
    const ModBaseDecoder decoder(configs);
    std::vector<DecodedChunk> chunks{
            make_chunk("ACGTACGTAACCGGTTAACGTACGTAACCGGTTAACGTACGTAACCGGTT"),
            make_chunk("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            make_chunk("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")};
    const auto scores = make_scores(std::ssize(chunks));

    std::vector<std::pair<std::vector<uint8_t>, std::vector<bool>>> expected;
    for (int64_t i = 0; i < std::ssize(chunks); ++i) {
        const std::vector<at::Tensor> heads{scores[i]};
        expected.push_back(decoder.decode(heads, chunks[i].sequence, chunks[i].moves));
    }

    decoder.decode(std::vector<at::Tensor>{scores}, chunks);

    for (size_t i = 0; i < chunks.size(); ++i) {
        CATCH_CHECK(chunks[i].base_mod_probs == expected[i].first);
        CATCH_CHECK(chunks[i].base_mod_simplex_motif_hits == expected[i].second);
    }
    // A chunk with no A has no motif hits; a chunk of all A has hits.
    CATCH_CHECK(chunks[2].base_mod_simplex_motif_hits ==
                std::vector<bool>(chunks[2].sequence.size(), false));
    CATCH_CHECK(chunks[1].base_mod_simplex_motif_hits != chunks[2].base_mod_simplex_motif_hits);
}

CATCH_TEST_CASE(CUT_TAG " pack_decoded_chunks uses the decoder tensor layout", CUT_TAG) {
    const std::vector<DecodedChunk> chunks{make_chunk("ACG"), make_chunk("TA")};
    const auto packed = pack_decoded_chunks(chunks);

    CATCH_REQUIRE(packed.sizes() == at::IntArrayRef({3, 2, T}));
    CATCH_REQUIRE(packed.scalar_type() == at::kChar);
    for (int64_t n = 0; n < 2; ++n) {
        for (int64_t t = 0; t < T; ++t) {
            CATCH_CHECK(packed[0][n][t].item<int8_t>() == static_cast<int8_t>(chunks[n].moves[t]));
        }
        for (size_t i = 0; i < chunks[n].sequence.size(); ++i) {
            CATCH_CHECK(packed[1][n][i].item<int8_t>() == chunks[n].sequence[i]);
        }
        // Remainder of the sequence row is zero padding, qstring is unused.
        CATCH_CHECK(packed[1][n][T - 1].item<int8_t>() == 0);
        CATCH_CHECK(packed[2][n].count_nonzero().item<int64_t>() == 0);
    }
}

CATCH_TEST_CASE(CUT_TAG " no modbase configs decodes to nothing", CUT_TAG) {
    const ModBaseDecoder decoder({});
    CATCH_CHECK(decoder.empty());
    const auto chunk = make_chunk("ACGT");
    const auto [probs, hits] = decoder.decode({}, chunk.sequence, chunk.moves);
    CATCH_CHECK(probs.empty());
    CATCH_CHECK(hits.empty());
}
