#include "TestUtils.h"
#include "hts_utils/fai_utils.h"
#include "secondary/architectures/model_torch_base.h"
#include "secondary/features/decoder_base.h"
#include "variant/variant_impl.h"

#include <ATen/ATen.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado::smallvar::tests {

#define TEST_GROUP "[VariantImpl]"

namespace {

class StubEncoder final : public secondary::EncoderBase {
public:
    struct ExpectedRegion {
        std::string ref_name;
        int64_t ref_start;
        int64_t ref_end;
        int32_t seq_id;
    };

    StubEncoder(ExpectedRegion expected_region,
                std::unordered_map<std::string, int32_t> expected_haplotags,
                kadayashi::varcall_result_t produce_haplotags_result)
            : m_expected_region{std::move(expected_region)},
              m_expected_haplotags{std::move(expected_haplotags)},
              m_produce_haplotags_result{std::move(produce_haplotags_result)} {}

    kadayashi::varcall_result_t produce_haplotags(const std::string& ref_name,
                                                  const int64_t ref_start,
                                                  const int64_t ref_end) override {
        if ((ref_name != m_expected_region.ref_name) ||
            (ref_start != m_expected_region.ref_start) || (ref_end != m_expected_region.ref_end)) {
            throw std::runtime_error{"Unexpected region passed to produce_haplotags."};
        }
        return m_produce_haplotags_result;
    }

    secondary::Sample encode_region(
            const std::string& ref_name,
            const int64_t ref_start,
            const int64_t ref_end,
            const int32_t seq_id,
            const std::unordered_map<std::string, int32_t>& haplotags) override {
        if ((ref_name != m_expected_region.ref_name) ||
            (ref_start != m_expected_region.ref_start) || (ref_end != m_expected_region.ref_end) ||
            (seq_id != m_expected_region.seq_id) || (haplotags != m_expected_haplotags)) {
            throw std::runtime_error{"Unexpected arguments passed to encode_region."};
        }
        return {};
    }

    at::Tensor collate(std::vector<at::Tensor> batch, const bool /*pinned_memory*/) const override {
        return std::empty(batch) ? at::empty({0}) : std::move(batch.front());
    }

    std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const override {
        return samples;
    }

    secondary::FeatureColumnMap get_feature_column_map() const override { return {}; }

private:
    ExpectedRegion m_expected_region;
    std::unordered_map<std::string, int32_t> m_expected_haplotags;
    kadayashi::varcall_result_t m_produce_haplotags_result;
};

class SequencedStubEncoder final : public secondary::EncoderBase {
public:
    struct RegionCall {
        StubEncoder::ExpectedRegion expected_region;
        std::unordered_map<std::string, int32_t> expected_haplotags;
        kadayashi::varcall_result_t produce_haplotags_result;
    };

    explicit SequencedStubEncoder(std::vector<RegionCall> calls) : m_calls{std::move(calls)} {}

    kadayashi::varcall_result_t produce_haplotags(const std::string& ref_name,
                                                  const int64_t ref_start,
                                                  const int64_t ref_end) override {
        if (m_next_call >= std::ssize(m_calls)) {
            throw std::runtime_error{"Unexpected extra produce_haplotags call."};
        }
        const RegionCall& call = m_calls[m_next_call];
        const StubEncoder::ExpectedRegion& expected_region = call.expected_region;
        if ((ref_name != expected_region.ref_name) || (ref_start != expected_region.ref_start) ||
            (ref_end != expected_region.ref_end)) {
            throw std::runtime_error{"Unexpected region passed to produce_haplotags."};
        }
        return call.produce_haplotags_result;
    }

    secondary::Sample encode_region(
            const std::string& ref_name,
            const int64_t ref_start,
            const int64_t ref_end,
            const int32_t seq_id,
            const std::unordered_map<std::string, int32_t>& haplotags) override {
        if (m_next_call >= std::ssize(m_calls)) {
            throw std::runtime_error{"Unexpected extra encode_region call."};
        }
        const RegionCall& call = m_calls[m_next_call];
        const StubEncoder::ExpectedRegion& expected_region = call.expected_region;
        if ((ref_name != expected_region.ref_name) || (ref_start != expected_region.ref_start) ||
            (ref_end != expected_region.ref_end) || (seq_id != expected_region.seq_id) ||
            (haplotags != call.expected_haplotags)) {
            throw std::runtime_error{"Unexpected arguments passed to encode_region."};
        }
        ++m_next_call;
        return {};
    }

    at::Tensor collate(std::vector<at::Tensor> batch, const bool /*pinned_memory*/) const override {
        return std::empty(batch) ? at::empty({0}) : std::move(batch.front());
    }

    std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const override {
        return samples;
    }

    secondary::FeatureColumnMap get_feature_column_map() const override { return {}; }

private:
    std::vector<RegionCall> m_calls;
    int64_t m_next_call{0};
};

class CollatingStubEncoder final : public secondary::EncoderBase {
public:
    kadayashi::varcall_result_t produce_haplotags(const std::string&,
                                                  const int64_t,
                                                  const int64_t) override {
        throw std::runtime_error{"Unexpected produce_haplotags call."};
    }

    secondary::Sample encode_region(const std::string&,
                                    const int64_t,
                                    const int64_t,
                                    const int32_t,
                                    const std::unordered_map<std::string, int32_t>&) override {
        throw std::runtime_error{"Unexpected encode_region call."};
    }

    at::Tensor collate(std::vector<at::Tensor> batch, const bool /*pinned_memory*/) const override {
        if (std::empty(batch)) {
            return at::empty({0}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        }
        return at::stack(batch);
    }

    std::vector<secondary::Sample> merge_adjacent_samples(
            std::vector<secondary::Sample> samples) const override {
        return samples;
    }

    secondary::FeatureColumnMap get_feature_column_map() const override { return {}; }
};

class StubModel final : public secondary::ModelTorchBase {
public:
    StubModel(const MustConstructWithFactory& ctor_tag,
              const double memory_scale,
              const float output_offset)
            : secondary::ModelTorchBase(ctor_tag),
              m_memory_scale{memory_scale},
              m_output_offset{output_offset} {}

    at::Tensor forward(at::Tensor x) override { return x + m_output_offset; }

    double estimate_batch_memory(const std::vector<int64_t>& batch_tensor_shape) const override {
        double estimated_elements = 1.0;
        for (const int64_t dim : batch_tensor_shape) {
            estimated_elements *= static_cast<double>(dim);
        }
        return estimated_elements * m_memory_scale;
    }

private:
    double m_memory_scale{1.0};
    float m_output_offset{0.0f};
};

std::unique_ptr<secondary::EncoderBase> make_haplotagging_encoder(
        const std::filesystem::path& in_ref_fn,
        const std::filesystem::path& in_bam_aln_fn) {
    const secondary::ModelConfig model_config{
            .feature_encoder_type = "ReadAlignmentFeatureEncoder",
            .feature_encoder_kwargs =
                    {
                            {"tag_keep_missing", "false"},
                            {"min_mapq", "1"},
                            {"max_reads", "100"},
                            {"row_per_read", "false"},
                            {"include_dwells", "true"},
                            {"include_haplotype", "true"},
                            {"include_snp_qv", "true"},
                            {"right_align_insertions", "false"},
                    },
            .feature_encoder_dtypes = {},
    };

    const std::string tag_name{};
    const int32_t tag_value{0};
    const std::string read_group{};
    const bool clip_to_zero{true};
    const double min_snp_accuracy{0.0};
    const secondary::HaplotagSource hap_source{secondary::HaplotagSource::COMPUTE};
    const std::optional<std::filesystem::path> phasing_bin{};
    const secondary::KadayashiOptions kadayashi_opt{
            .max_clipping = 100000,
            .min_strand_cov = 1,
    };

    return secondary::encoder_factory(model_config, in_ref_fn, in_bam_aln_fn, read_group, tag_name,
                                      tag_value, clip_to_zero, min_snp_accuracy, std::nullopt,
                                      std::nullopt, hap_source, phasing_bin, kadayashi_opt, false);
}

secondary::Sample make_sample(const int32_t seq_id,
                              const std::vector<int64_t>& positions_major,
                              const float feature_offset) {
    const at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const int64_t num_positions = std::ssize(positions_major);
    constexpr int64_t DEPTH = 1;
    constexpr int64_t NUM_FEATURES = 2;

    secondary::Sample sample;
    sample.seq_id = seq_id;
    // Make a minimal read-level tensor with shape [positions, depth, features].
    sample.features = (at::arange(num_positions * DEPTH * NUM_FEATURES, options)
                               .reshape({num_positions, DEPTH, NUM_FEATURES}) +
                       feature_offset);
    sample.positions_major = positions_major;
    sample.positions_minor = std::vector<int64_t>(std::size(positions_major), 0);
    sample.depth = at::ones({num_positions}, options);

    return sample;
}

secondary::Sample make_zero_depth_sample(const int32_t seq_id,
                                         const std::vector<int64_t>& positions_major) {
    const at::TensorOptions options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    const int64_t num_positions = std::ssize(positions_major);
    constexpr int64_t NUM_FEATURES = 2;

    secondary::Sample sample;
    sample.seq_id = seq_id;
    sample.features = at::empty({num_positions, 0, NUM_FEATURES}, options);
    sample.positions_major = positions_major;
    sample.positions_minor = std::vector<int64_t>(std::size(positions_major), 0);
    sample.depth = at::zeros({num_positions}, options);

    return sample;
}

at::Tensor make_haploid_probs(const std::string_view symbols,
                              const std::string_view seq,
                              const float tp_prob) {
    if (std::empty(seq)) {
        return at::empty({0, 0}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    }

    // Create a symbol lookup.
    std::array<int32_t, 256> lookup{};
    lookup.fill(static_cast<int32_t>(std::size(symbols)));
    for (int32_t i = 0; i < static_cast<int32_t>(std::size(symbols)); ++i) {
        lookup[static_cast<int32_t>(symbols[i])] = i;
    }

    const float fp_prob = (1.0f - tp_prob) / static_cast<float>(std::ssize(symbols) - 1);

    at::Tensor probs = at::full({std::ssize(seq), std::ssize(symbols)}, fp_prob,
                                at::TensorOptions().dtype(at::kFloat).device(at::kCPU));

    for (int64_t row = 0; row < std::ssize(seq); ++row) {
        const int64_t col = lookup[static_cast<int32_t>(seq[row])];
        probs.index_put_({row, col}, tp_prob);
    }

    return probs;
}

at::Tensor make_polyploid_probs(const std::string_view symbols,
                                const std::vector<std::string_view>& cons_seqs,
                                const std::vector<float>& true_pos_probs) {
    if (std::empty(cons_seqs)) {
        return at::empty({0, 0, 0}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    }

    if (std::size(cons_seqs) != std::size(true_pos_probs)) {
        throw std::runtime_error{"Expected one true-position probability per haplotype."};
    }

    const size_t len = std::size(cons_seqs.front());
    for (const std::string_view seq : cons_seqs) {
        if (std::size(seq) != len) {
            throw std::runtime_error("All input sequences need to be of the same length! len: " +
                                     std::to_string(len) +
                                     ", found: " + std::to_string(std::size(seq)));
        }
    }

    // Fill the probabilities for the input sequences.
    std::vector<at::Tensor> all_probs;
    all_probs.reserve(std::size(cons_seqs));
    for (size_t i = 0; i < std::size(cons_seqs); ++i) {
        all_probs.emplace_back(make_haploid_probs(symbols, cons_seqs[i], true_pos_probs[i]));
    }

    return at::stack(all_probs, /*dim=*/1);
}

}  // namespace

CATCH_TEST_CASE("variant_impl workflow worker functions", TEST_GROUP) {
    /**
     * \brief Covers small worker-facing helpers that are easier to validate in isolation than
     *          through the full async pipeline.
     *
     *          The sections below check three things:
     *          signal_worker_terminate flips the shared stop flag, create_resources rejects
     *          unsupported devices before model setup begins, and convert_variants produces the
     *          normalized variant records expected by the rest of the workflow, including a few
     *          legacy edge cases preserved from the older haplotagging path.
     */
    CATCH_SECTION("signal_worker_terminate sets the shared termination flag") {
        std::atomic<bool> worker_terminate{false};

        signal_worker_terminate(worker_terminate);

        CATCH_CHECK(worker_terminate.load(std::memory_order_acquire));
    }

    CATCH_SECTION("create_resources rejects unsupported devices before model initialization") {
        const secondary::ModelConfig model_config{};

        CATCH_REQUIRE_THROWS_WITH(
                create_resources(model_config, std::filesystem::path{}, std::filesystem::path{},
                                 "metal", 1, 1, true, "", "", 0, 0.0, std::nullopt, std::nullopt,
                                 std::nullopt, std::nullopt, {}, false),
                Catch::Matchers::ContainsSubstring("Unsupported device: metal"));
    }

    CATCH_SECTION("convert_variants normalizes diploid genotypes and filters") {
        const std::vector<kadayashi::variant_dorado_style_t> kadayashi_variants{
                {
                        .is_confident = true,
                        .is_phased = true,
                        .pos = 4,
                        .qual = 60,
                        .ref = "A",
                        .alts = {"T"},
                        .genotype = {'0', '1'},
                        .min_depth = 10,
                        .median_depth = 10,
                },
                {
                        .is_confident = false,
                        .is_phased = false,
                        .pos = 8,
                        .qual = 10,
                        .ref = "C",
                        .alts = {"G"},
                        .genotype = {'1', '1'},
                        .min_depth = 1,
                        .median_depth = 1,
                },
                {
                        .is_confident = true,
                        .is_phased = true,
                        .pos = 12,
                        .qual = 50,
                        .ref = "A",
                        .alts = {"T", "C"},
                        .genotype = {'1', '2'},
                        .min_depth = 15,
                        .median_depth = 15,
                },
        };

        // clang-format off
        const std::vector<secondary::Variant> expected{
                secondary::Variant{7, 4, "A", {"T"}, "PASS", {}, 60.0f,
                                   {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "10"}}, 0, 0},
                secondary::Variant{7, 8, "C", {"G"}, "LowQual", {}, 10.0f,
                                   {{"GT", "1/1"}, {"GQ", "10"}, {"DP", "1"}}, 0, 0},
                secondary::Variant{7, 12, "A", {"C", "T"}, "PASS", {}, 50.0f,
                                   {{"GT", "1/2"}, {"GQ", "50"}, {"DP", "15"}}, 0, 0},
        };
        // clang-format on

        const std::vector<secondary::Variant> variants =
                convert_variants(kadayashi_variants, 7, 2, 30.0f);

        CATCH_CHECK(variants == expected);
    }

    CATCH_SECTION("convert_variants preserves legacy haplotagging edge cases") {
        struct TestCase {
            std::string name;
            std::vector<kadayashi::variant_dorado_style_t> kadayashi_variants;
            int32_t seq_id;
            int32_t ploidy;
            float pass_min_qual;
            std::vector<secondary::Variant> expected;
        };

        // clang-format off
        const std::vector<TestCase> test_cases{
            TestCase{"Empty input", {}, 0, 2, 3.0f, {}},
            TestCase{
                "Single hom variant",
                {
                    kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {"T"}, {1, 1}, 7, 7},
                },
                123, 2, 3.0f,
                {
                    secondary::Variant{123, 5, "A", {"T"}, "PASS", {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "7"}}, 0, 0},
                },
            },
            TestCase{
                "Single het variant where one alt matches the ref",
                {
                    kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {"T"}, {0, 1}, 20, 20},
                },
                123, 2, 3.0f,
                {
                    secondary::Variant{123, 5, "A", {"T"}, "PASS", {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}, 0, 0},
                },
            },
            TestCase{
                "Single het variant where both alts differ from ref",
                {
                    kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {"T", "C"}, {0, 1}, 9, 10},
                },
                123, 2, 3.0f,
                {
                    secondary::Variant{123, 5, "A", {"C", "T"}, "PASS", {}, 60.0f, {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "10"}}, 0, 0},
                },
            },
            TestCase{
                "Malformed input without alts falls back to a filtered placeholder",
                {
                    kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {}, {0, 1}, 4, 4},
                },
                123, 2, 3.0f,
                {
                    secondary::Variant{123, 5, "A", {"."}, ".", {}, 60.0f, {{"GT", "0/0"}, {"GQ", "60"}, {"DP", "4"}}, 0, 0},
                },
            },
            TestCase{
                "Confidence and phasing flags are ignored during conversion",
                {
                    kadayashi::variant_dorado_style_t{false, false, 5, 60, "A", {"T"}, {1, 1}, 15, 15},
                },
                123, 2, 3.0f,
                {
                    secondary::Variant{123, 5, "A", {"T"}, "PASS", {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "15"}}, 0, 0},
                },
            },
        };
        // clang-format on

        for (const auto& test_case : test_cases) {
            CATCH_CAPTURE(test_case.name);

            const std::vector<secondary::Variant> result =
                    convert_variants(test_case.kadayashi_variants, test_case.seq_id,
                                     test_case.ploidy, test_case.pass_min_qual);

            CATCH_CHECK(result == test_case.expected);
        }
    }
}

CATCH_TEST_CASE("batch and inference workflow functions operate on synthetic samples", TEST_GROUP) {
    /**
     * \brief Exercises the queueing, batching, and decode hand-off stages with small synthetic
     *          samples instead of full input fixtures.
     *
     *          The goal here is to validate workflow mechanics rather than model accuracy:
     *          - worker_batch_producer should isolate odd-sized inputs into their own batch
     *          - worker_infer_samples_in_parallel should collate the sample tensors and forward model
     *            outputs unchanged except for the stubbed model offset
     *          - worker_separate_decode_data should split the batched logits back into one work item per sample.
     */
    const int32_t window_len = 4;
    const secondary::Sample sample_a = make_sample(0, {0, 1, 2, 3}, 0.0f);
    const secondary::Sample sample_b = make_sample(0, {10, 11, 12, 13}, 10.0f);
    const secondary::Sample odd_sample = make_sample(0, {20, 21, 22}, 20.0f);
    const secondary::Sample zero_depth_sample = make_zero_depth_sample(0, {30, 31, 32, 33});

    CATCH_SECTION("worker_batch_producer isolates odd samples and preserves fixed-size batches") {
        auto model = secondary::ModelTorchBase::make<StubModel>(1.0, 0.0f);

        utils::AsyncQueue<InferenceData> input_queue{2};
        utils::AsyncQueue<InferenceData> output_queue{4};

        // Input samples are a combination of 2 full-length samples, and one short sample (middle one).
        InferenceData input_batch;
        input_batch.samples = {sample_a, odd_sample, sample_b};
        CATCH_REQUIRE(input_queue.try_push(std::move(input_batch)) ==
                      utils::AsyncQueueStatus::Success);
        input_queue.terminate(utils::AsyncQueueTerminateFast::No);

        // Workflow control variables.
        std::atomic<bool> worker_terminate{false};
        secondary::WorkerReturnStatus ret_status;

        // Run unit under test.
        worker_batch_producer(input_queue, output_queue, worker_terminate, ret_status, *model,
                              window_len, /*batch_size=*/2, /*max_available_mem=*/1024.0, false);

        // Test successful run.
        CATCH_REQUIRE(!ret_status.exception_thrown);
        CATCH_REQUIRE(!worker_terminate.load());
        CATCH_REQUIRE(std::size(output_queue) == 2);

        // Check that the first batch contains 1 element - the odd one.
        InferenceData first_output;
        CATCH_REQUIRE(output_queue.try_pop(first_output) == utils::AsyncQueueStatus::Success);
        CATCH_REQUIRE(std::size(first_output.samples) == 1);
        CATCH_CHECK(first_output.samples[0].positions_major == odd_sample.positions_major);

        // Second batch should contain 2 elements: sample_a and sample_b.
        InferenceData second_output;
        CATCH_REQUIRE(output_queue.try_pop(second_output) == utils::AsyncQueueStatus::Success);
        CATCH_REQUIRE(std::size(second_output.samples) == 2);
        CATCH_CHECK(second_output.samples[0].positions_major == sample_a.positions_major);
        CATCH_CHECK(second_output.samples[1].positions_major == sample_b.positions_major);

        // Nothing else in the queue.
        InferenceData exhausted_output;
        CATCH_CHECK(output_queue.try_pop(exhausted_output) == utils::AsyncQueueStatus::Terminate);
    }

    CATCH_SECTION("worker_batch_producer skips samples with coordinates but zero feature rows") {
        auto model = secondary::ModelTorchBase::make<StubModel>(1.0, 0.0f);

        utils::AsyncQueue<InferenceData> input_queue{2};
        utils::AsyncQueue<InferenceData> output_queue{4};

        InferenceData input_batch;
        input_batch.samples = {sample_a, zero_depth_sample, sample_b};
        CATCH_REQUIRE(input_queue.try_push(std::move(input_batch)) ==
                      utils::AsyncQueueStatus::Success);
        input_queue.terminate(utils::AsyncQueueTerminateFast::No);

        std::atomic<bool> worker_terminate{false};
        secondary::WorkerReturnStatus ret_status;

        worker_batch_producer(input_queue, output_queue, worker_terminate, ret_status, *model,
                              window_len, /*batch_size=*/2, /*max_available_mem=*/1024.0, false);

        CATCH_REQUIRE(!ret_status.exception_thrown);
        CATCH_REQUIRE(!worker_terminate.load());
        CATCH_REQUIRE(std::size(output_queue) == 1);

        InferenceData output_batch;
        CATCH_REQUIRE(output_queue.try_pop(output_batch) == utils::AsyncQueueStatus::Success);
        CATCH_REQUIRE(std::size(output_batch.samples) == 2);
        CATCH_CHECK(output_batch.samples[0].positions_major == sample_a.positions_major);
        CATCH_CHECK(output_batch.samples[1].positions_major == sample_b.positions_major);

        InferenceData exhausted_output;
        CATCH_CHECK(output_queue.try_pop(exhausted_output) == utils::AsyncQueueStatus::Terminate);
    }

    CATCH_SECTION("worker_infer_samples_in_parallel collates batch tensors and forwards logits") {
        ////////////////////////////////////////
        /// Set up the context for the test. ///
        ////////////////////////////////////////
        constexpr float OUTPUT_OFFSET = 3.5f;
        auto model = secondary::ModelTorchBase::make<StubModel>(1.0, OUTPUT_OFFSET);
        model->set_normalise(false);

        std::vector<std::shared_ptr<secondary::ModelTorchBase>> models;
        models.emplace_back(model);

        std::vector<std::unique_ptr<secondary::EncoderBase>> encoders;
        encoders.emplace_back(std::make_unique<CollatingStubEncoder>());

        const std::vector<c10::optional<c10::Stream>> streams{std::nullopt};
        const std::vector<std::pair<std::string, int64_t>> draft_lens{{"chr1", 100}};

        utils::AsyncQueue<InferenceData> batch_queue{2};
        utils::AsyncQueue<DecodeData> decode_queue{2};

        InferenceData batch;
        batch.samples = {sample_a, sample_b};
        CATCH_REQUIRE(batch_queue.try_push(std::move(batch)) == utils::AsyncQueueStatus::Success);
        batch_queue.terminate(utils::AsyncQueueTerminateFast::No);

        std::atomic<bool> worker_terminate{false};
        secondary::WorkerReturnStatus ret_status;

        ////////////////////////////////////////
        /// Run the unit under test.         ///
        ////////////////////////////////////////
        worker_infer_samples_in_parallel(batch_queue, decode_queue, models, worker_terminate,
                                         ret_status, streams, encoders, draft_lens, false);

        ////////////////////////////////////////
        /// Eval.                            ///
        ////////////////////////////////////////
        CATCH_REQUIRE(!ret_status.exception_thrown);
        CATCH_REQUIRE(!worker_terminate.load());
        CATCH_REQUIRE(std::size(decode_queue) == 1);

        DecodeData decoded_batch;
        CATCH_REQUIRE(decode_queue.try_pop(decoded_batch) == utils::AsyncQueueStatus::Success);
        CATCH_REQUIRE(std::size(decoded_batch.samples) == 2);
        CATCH_CHECK(decoded_batch.samples[0].positions_major == sample_a.positions_major);
        CATCH_CHECK(decoded_batch.samples[1].positions_major == sample_b.positions_major);

        const at::Tensor expected_logits =
                at::stack(std::vector<at::Tensor>{sample_a.features, sample_b.features}) +
                OUTPUT_OFFSET;
        CATCH_CHECK(at::equal(decoded_batch.logits, expected_logits));

        DecodeData exhausted_decode_data;
        CATCH_CHECK(decode_queue.try_pop(exhausted_decode_data) ==
                    utils::AsyncQueueStatus::Terminate);
    }

    CATCH_SECTION("worker_separate_decode_data splits batched logits into per-sample work items") {
        ////////////////////////////////////////
        /// Set up the context for the test. ///
        ////////////////////////////////////////
        const at::Tensor batched_logits = at::stack(
                std::vector<at::Tensor>{sample_a.features + 1.0f, sample_b.features + 2.0f});

        utils::AsyncQueue<DecodeData> input_queue{2};
        utils::AsyncQueue<secondary::VariantCallingSample> output_queue{4};

        DecodeData decode_data;
        decode_data.samples = {sample_a, sample_b};
        decode_data.logits = batched_logits;
        CATCH_REQUIRE(input_queue.try_push(std::move(decode_data)) ==
                      utils::AsyncQueueStatus::Success);
        input_queue.terminate(utils::AsyncQueueTerminateFast::No);

        std::atomic<bool> worker_terminate{false};
        secondary::WorkerReturnStatus ret_status;

        ////////////////////////////////////////
        /// Run the unit under test.         ///
        ////////////////////////////////////////
        worker_separate_decode_data(input_queue, output_queue, worker_terminate, ret_status,
                                    /*num_threads=*/1, /*continue_on_exception=*/false);

        ////////////////////////////////////////
        /// Eval.                            ///
        ////////////////////////////////////////
        CATCH_REQUIRE(!ret_status.exception_thrown);
        CATCH_REQUIRE(!worker_terminate.load());
        CATCH_REQUIRE(std::size(output_queue) == 2);

        secondary::VariantCallingSample first_sample;
        CATCH_REQUIRE(output_queue.try_pop(first_sample) == utils::AsyncQueueStatus::Success);
        CATCH_CHECK(first_sample.seq_id == sample_a.seq_id);
        CATCH_CHECK(first_sample.positions_major == sample_a.positions_major);
        CATCH_CHECK(first_sample.positions_minor == sample_a.positions_minor);
        CATCH_CHECK(at::equal(first_sample.logits, batched_logits[0]));

        secondary::VariantCallingSample second_sample;
        CATCH_REQUIRE(output_queue.try_pop(second_sample) == utils::AsyncQueueStatus::Success);
        CATCH_CHECK(second_sample.seq_id == sample_b.seq_id);
        CATCH_CHECK(second_sample.positions_major == sample_b.positions_major);
        CATCH_CHECK(second_sample.positions_minor == sample_b.positions_minor);
        CATCH_CHECK(at::equal(second_sample.logits, batched_logits[1]));

        secondary::VariantCallingSample exhausted_variant_calling_sample;
        CATCH_CHECK(output_queue.try_pop(exhausted_variant_calling_sample) ==
                    utils::AsyncQueueStatus::Terminate);
    }
}

CATCH_TEST_CASE("worker_sample_producer preserves simple pass variants when encode_region is empty",
                TEST_GROUP) {
    /**
     * \brief Verifies that worker_sample_producer still preserves simple PASS variants in the unique
     *          window span even when no inference sample is emitted for the BAM window.
     *
     *          The stub encoder returns confident simple variants but an empty encoded region.
     *          The expected result is that nothing is pushed to the inference queue, the BAM
     *          window is still accounted for in the reduction state, and the converted simple
     *          variant in the window's unique span is accumulated in
     *          chrom_reduce_data[0].variants_simple.
     */

    ////////////////////////////////////////
    /// Set up the context for the test. ///
    ////////////////////////////////////////
    const secondary::Window bam_window{
            .seq_id = 0,
            .seq_length = 10,
            .start = 0,
            .end = 10,
            .start_no_overlap = 2,
            .end_no_overlap = 8,
            .source_region_id = 0,
    };

    const auto make_simple_variant = [](const uint32_t pos) {
        return kadayashi::variant_dorado_style_t{
                .is_confident = true,
                .is_phased = true,
                .pos = pos,
                .qual = 60,
                .ref = "A",
                .alts = {"T"},
                .genotype = {'1', '1'},
                .median_depth = -1,
        };
    };
    const kadayashi::variant_dorado_style_t left_overlap_variant = make_simple_variant(1);
    const kadayashi::variant_dorado_style_t simple_variant = make_simple_variant(4);
    const kadayashi::variant_dorado_style_t right_boundary_variant = make_simple_variant(8);

    const std::vector<secondary::Variant> expected_variants =
            convert_variants({simple_variant}, bam_window.seq_id, 2, 30.0f);

    const std::vector<std::vector<secondary::Window>> bam_regions{{bam_window}};
    const std::vector<std::pair<std::string, int64_t>> draft_lens{{"chr1", bam_window.seq_length}};
    const std::vector<std::string> draft_seqs(std::size(draft_lens));

    // Mock encoder which returns zero samples for this region.
    VariantResources resources;
    resources.encoders.emplace_back(std::make_unique<StubEncoder>(
            StubEncoder::ExpectedRegion{"chr1", bam_window.start, bam_window.end,
                                        bam_window.seq_id},
            std::unordered_map<std::string, int32_t>{{"read-1", 1}},
            kadayashi::varcall_result_t{
                    .qname2hp = {{"read-1", 0}},
                    .variants = {left_overlap_variant, simple_variant, right_boundary_variant},
                    .phasing_breakpoints = {},
            }));
    auto model = secondary::ModelTorchBase::make<StubModel>(1.0, 0.0f);
    resources.models.emplace_back(model);

    // Initialize the data for reduction, updated by worker_sample_producer.
    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    chrom_reduce_data[0].seq_id = bam_window.seq_id;
    chrom_reduce_data[0].seq_name = "chr1";
    chrom_reduce_data[0].seq_len = bam_window.seq_length;
    chrom_reduce_data[0].num_bam_regions = 1;
    chrom_reduce_data[0].remaining_bam_regions = 1;

    // Populate the input queue.
    utils::AsyncQueue<secondary::Window> input_queue{2};
    CATCH_REQUIRE(input_queue.try_push(secondary::Window{bam_window}) ==
                  utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Create the output queue.
    utils::AsyncQueue<InferenceData> output_queue{2};

    // Workflow control parameters.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    // Tracker statistics.
    secondary::Stats stats;
    stats.set("processed", 0.0);

    ////////////////////////////////////////
    /// Run the unit under test.         ///
    ////////////////////////////////////////
    worker_sample_producer(input_queue, output_queue, chrom_reduce_data, resources, stats,
                           worker_terminate, ret_status, bam_regions, draft_lens, draft_seqs,
                           secondary::VariantCandidateSource::COMPUTE, std::nullopt, 1,
                           bam_window.seq_length, 0, 10, false, 2, 30.0f, false, false, 0, 0, 0.0f,
                           0);

    ////////////////////////////////////////
    /// Eval.                            ///
    ////////////////////////////////////////
    CATCH_CHECK(!ret_status.exception_thrown);
    CATCH_CHECK(!worker_terminate.load());
    CATCH_CHECK(output_queue.size() == 0);
    CATCH_CHECK(chrom_reduce_data[0].ready);
    CATCH_CHECK(chrom_reduce_data[0].remaining_bam_regions == 0);
    CATCH_CHECK(chrom_reduce_data[0].num_samples == 0);
    CATCH_CHECK(chrom_reduce_data[0].variants_simple == expected_variants);
    CATCH_CHECK(stats.get_stats().at("processed") == 6.0);
}

CATCH_TEST_CASE("worker_sample_producer ignores simple variants outside a window's unique span",
                TEST_GROUP) {
    /**
     * \brief Reproduces duplicate Kadayashi PASS variant accumulation from overlapping BAM
     *          windows.
     *
     *          The same simple variant is returned for two overlapping windows. The first window's
     *          no-overlap span owns the variant coordinate; the second only sees it through its
     *          overlap with the first. The producer should only accumulate the owner window's
     *          record.
     */
    const secondary::Window first_window{
            .seq_id = 0,
            .seq_length = 200,
            .start = 0,
            .end = 100,
            .start_no_overlap = 0,
            .end_no_overlap = 100,
            .source_region_id = 0,
    };
    const secondary::Window second_window{
            .seq_id = 0,
            .seq_length = 200,
            .start = 60,
            .end = 160,
            .start_no_overlap = 100,
            .end_no_overlap = 160,
            .source_region_id = 0,
    };

    const kadayashi::variant_dorado_style_t overlapping_variant{
            .is_confident = true,
            .is_phased = true,
            .pos = 90,
            .qual = 60,
            .ref = "A",
            .alts = {"T"},
            .genotype = {'0', '1'},
            .median_depth = -1,
    };

    const std::vector<secondary::Variant> expected_variants =
            convert_variants({overlapping_variant}, first_window.seq_id, 2, 30.0f);

    const std::vector<std::vector<secondary::Window>> bam_regions{
            {first_window, second_window},
    };
    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", first_window.seq_length},
    };
    const std::vector<std::string> draft_seqs(std::size(draft_lens));

    VariantResources resources;
    resources.encoders.emplace_back(
            std::make_unique<SequencedStubEncoder>(std::vector<SequencedStubEncoder::RegionCall>{
                    {
                            .expected_region = {"chr1", first_window.start, first_window.end,
                                                first_window.seq_id},
                            .expected_haplotags = {{"read-1", 1}},
                            .produce_haplotags_result =
                                    kadayashi::varcall_result_t{
                                            .qname2hp = {{"read-1", 0}},
                                            .variants = {overlapping_variant},
                                            .phasing_breakpoints = {},
                                    },
                    },
                    {
                            .expected_region = {"chr1", second_window.start, second_window.end,
                                                second_window.seq_id},
                            .expected_haplotags = {{"read-2", 1}},
                            .produce_haplotags_result =
                                    kadayashi::varcall_result_t{
                                            .qname2hp = {{"read-2", 0}},
                                            .variants = {overlapping_variant},
                                            .phasing_breakpoints = {},
                                    },
                    },
            }));
    auto model = secondary::ModelTorchBase::make<StubModel>(1.0, 0.0f);
    resources.models.emplace_back(model);

    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    chrom_reduce_data[0].seq_id = first_window.seq_id;
    chrom_reduce_data[0].seq_name = "chr1";
    chrom_reduce_data[0].seq_len = first_window.seq_length;
    chrom_reduce_data[0].num_bam_regions = 2;
    chrom_reduce_data[0].remaining_bam_regions = 2;

    utils::AsyncQueue<secondary::Window> input_queue{4};
    CATCH_REQUIRE(input_queue.try_push(secondary::Window{first_window}) ==
                  utils::AsyncQueueStatus::Success);
    CATCH_REQUIRE(input_queue.try_push(secondary::Window{second_window}) ==
                  utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    utils::AsyncQueue<InferenceData> output_queue{2};

    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    secondary::Stats stats;
    stats.set("processed", 0.0);

    worker_sample_producer(input_queue, output_queue, chrom_reduce_data, resources, stats,
                           worker_terminate, ret_status, bam_regions, draft_lens, draft_seqs,
                           secondary::VariantCandidateSource::COMPUTE, std::nullopt, 1,
                           /*window_len=*/100, /*window_overlap=*/40,
                           /*variant_flanking_bases=*/10, /*continue_on_exception=*/false,
                           /*ploidy=*/2, /*pass_min_qual=*/30.0f,
                           /*tiled_regions=*/false, /*tiled_ext_flanks=*/false,
                           /*tiled_ext_major=*/0, /*tiled_ext_min_cov=*/0,
                           /*tiled_ext_cov_fract=*/0.0f, /*min_depth=*/0);

    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());
    CATCH_CHECK(output_queue.size() == 0);
    CATCH_CHECK(chrom_reduce_data[0].ready);
    CATCH_CHECK(chrom_reduce_data[0].remaining_bam_regions == 0);
    CATCH_CHECK(chrom_reduce_data[0].num_samples == 0);
    CATCH_CHECK(chrom_reduce_data[0].variants_simple == expected_variants);
}

CATCH_TEST_CASE("worker_sample_producer rejects bam windows with seq_id outside draft_lens",
                TEST_GROUP) {
    /**
     * \brief Verifies that worker_sample_producer rejects BAM windows whose seq_id cannot index
     *          draft_lens.
     *
     *          This protects the worker from reading the reference-name table out of bounds.
     *          The expected result is a reported worker exception, a termination signal, and an
     *          error message explaining that the seq_id is out of bounds for draft_lens.
     */

    ////////////////////////////////////////
    /// Set up the context for the test. ///
    ////////////////////////////////////////
    const secondary::Window bam_window{
            .seq_id = 1,  // <- seq_id > std::size(draft_lens) (below)
            .seq_length = 10,
            .start = 0,
            .end = 10,
            .start_no_overlap = 0,
            .end_no_overlap = 10,
            .source_region_id = 0,
    };

    // Mock encoder which returns zero samples for this region.
    VariantResources resources;
    resources.encoders.emplace_back(std::make_unique<StubEncoder>(
            StubEncoder::ExpectedRegion{"chr1", bam_window.start, bam_window.end,
                                        bam_window.seq_id},
            std::unordered_map<std::string, int32_t>{}, kadayashi::varcall_result_t{}));
    auto model = secondary::ModelTorchBase::make<StubModel>(1.0, 0.0f);
    resources.models.emplace_back(model);

    // Initialize the data for reduction, updated by worker_sample_producer.
    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    const std::vector<std::vector<secondary::Window>> bam_regions{{bam_window}};

    // Define the draft lens. Only one sequence.
    const std::vector<std::pair<std::string, int64_t>> draft_lens{{"chr1", 10}};
    const std::vector<std::string> draft_seqs(std::size(draft_lens));

    // Populate the input queue.
    utils::AsyncQueue<secondary::Window> input_queue{2};
    CATCH_REQUIRE(input_queue.try_push(secondary::Window{bam_window}) ==
                  utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Create the output queue.
    utils::AsyncQueue<InferenceData> output_queue{2};

    // Workflow control parameters.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    // Tracker statistics.
    secondary::Stats stats;
    stats.set("processed", 0.0);

    ////////////////////////////////////////
    /// Run the unit under test.         ///
    ////////////////////////////////////////
    worker_sample_producer(input_queue, output_queue, chrom_reduce_data, resources, stats,
                           worker_terminate, ret_status, bam_regions, draft_lens, draft_seqs,
                           secondary::VariantCandidateSource::COMPUTE, std::nullopt, 1,
                           bam_window.seq_length, 0, 10, false, 2, 30.0f, false, false, 0, 0, 0.0f,
                           0);

    ////////////////////////////////////////
    /// Eval.                            ///
    ////////////////////////////////////////
    CATCH_REQUIRE(ret_status.exception_thrown);
    CATCH_CHECK(worker_terminate.load());
    CATCH_CHECK(ret_status.message.find("out of bounds for draft_lens") != std::string::npos);
}

CATCH_TEST_CASE("worker_sample_producer rejects negative remaining_bam_regions", TEST_GROUP) {
    /**
     * \brief Verifies that worker_sample_producer detects malformed reduction bookkeeping when
     *          remaining_bam_regions becomes negative.
     *
     *          The test seeds the chromosome state so that decrementing the outstanding BAM-window
     *          count immediately underflows it. A negative value means the async workflow state is
     *          inconsistent, so the expected result is a worker failure, termination signaling, and
     *          an error message mentioning remaining_bam_regions.
     */
    ////////////////////////////////////////
    /// Set up the context for the test. ///
    ////////////////////////////////////////
    const secondary::Window bam_window{
            .seq_id = 0,
            .seq_length = 10,
            .start = 0,
            .end = 10,
            .start_no_overlap = 0,
            .end_no_overlap = 10,
            .source_region_id = 0,
    };

    // Mock encoder which returns zero samples for this region.
    VariantResources resources;
    resources.encoders.emplace_back(std::make_unique<StubEncoder>(
            StubEncoder::ExpectedRegion{"chr1", bam_window.start, bam_window.end,
                                        bam_window.seq_id},
            std::unordered_map<std::string, int32_t>{{"read-1", 1}},
            kadayashi::varcall_result_t{
                    .qname2hp = {{"read-1", 0}},
                    .variants = {},
                    .phasing_breakpoints = {},
            }));
    auto model = secondary::ModelTorchBase::make<StubModel>(1.0, 0.0f);
    resources.models.emplace_back(model);

    // Initialize the data for reduction, updated by worker_sample_producer.
    // The num_bam_regions and remaining_bam_regions are initialized to zero,
    // so when remaining_bam_regions is decremented in worker_sample_producer,
    // it becomes negative and should terminate.
    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    chrom_reduce_data[0].seq_id = bam_window.seq_id;
    chrom_reduce_data[0].seq_name = "chr1";
    chrom_reduce_data[0].seq_len = bam_window.seq_length;
    chrom_reduce_data[0].num_bam_regions = 0;
    chrom_reduce_data[0].remaining_bam_regions = 0;

    // Bam regions and draft lengths.
    const std::vector<std::vector<secondary::Window>> bam_regions{{bam_window}};
    const std::vector<std::pair<std::string, int64_t>> draft_lens{{"chr1", 10}};
    const std::vector<std::string> draft_seqs(std::size(draft_lens));

    // Populate the input queue.
    utils::AsyncQueue<secondary::Window> input_queue{2};
    CATCH_REQUIRE(input_queue.try_push(secondary::Window{bam_window}) ==
                  utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Create the output queue.
    utils::AsyncQueue<InferenceData> output_queue{2};

    // Workflow control parameters.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    // Tracker statistics.
    secondary::Stats stats;
    stats.set("processed", 0.0);

    ////////////////////////////////////////
    /// Run the unit under test.         ///
    ////////////////////////////////////////
    worker_sample_producer(input_queue, output_queue, chrom_reduce_data, resources, stats,
                           worker_terminate, ret_status, bam_regions, draft_lens, draft_seqs,
                           secondary::VariantCandidateSource::COMPUTE, std::nullopt, 1,
                           bam_window.seq_length, 0, 10, false, 2, 30.0f, false, false, 0, 0, 0.0f,
                           0);

    ////////////////////////////////////////
    /// Eval.                            ///
    ////////////////////////////////////////
    CATCH_REQUIRE(ret_status.exception_thrown);
    CATCH_CHECK(worker_terminate.load());
    CATCH_CHECK(ret_status.message.find("remaining_bam_regions") != std::string::npos);
}

CATCH_TEST_CASE("worker_sample_producer handles migrated haplotagging with real encoders",
                TEST_GROUP) {
    /**
     * \brief Exercises the migrated real-encoder haplotagging path through
     *          worker_sample_producer.
     *
     * Summary:
     * - Loads the real BAM and reference from test-02-supertiny.
     * - Uses real EncoderReadAlignment instances with HaplotagSource::COMPUTE.
     * - Runs worker_sample_producer on three BAM windows with VariantCandidateSource::COMPUTE.
     * - Checks that processing completes, the chromosome reduce state is marked ready, and all
     *   windows are consumed.
     * - Verifies that the expected simple PASS variants are accumulated in the reduce state.
     * - Verifies that any emitted inference items are structurally sane and that their count
     *   matches reduce_data.num_samples.
     */

    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_bam_aln_fn = test_data_dir / "in.aln.bam";
    const std::filesystem::path in_ref_fn = test_data_dir / "in.ref.fasta.gz";

    const std::vector<std::pair<std::string, int64_t>> draft_lens =
            utils::load_seq_lengths(in_ref_fn);

    const std::vector<std::string> draft_seqs(std::size(draft_lens));

    // Define input BAM regions for processing.
    const std::vector<secondary::Window> bam_windows{
            secondary::Window{0, 10000, 0, 300, 0, 300, -1},
            secondary::Window{0, 10000, 1000, 1800, 1000, 1800, -1},
            secondary::Window{0, 10000, 7000, 7500, 7000, 7500, -1},
    };
    const std::vector<std::vector<secondary::Window>> bam_regions{bam_windows};

    // Create the encoders.
    VariantResources resources;
    resources.encoders.emplace_back(make_haplotagging_encoder(in_ref_fn, in_bam_aln_fn));
    resources.encoders.emplace_back(make_haplotagging_encoder(in_ref_fn, in_bam_aln_fn));
    auto model = secondary::ModelTorchBase::make<StubModel>(1.0, 0.0f);
    resources.models.emplace_back(model);

    // Create the reduction information updated by worker_sample_producer.
    std::vector<ChromosomeReduceData> chrom_reduce_data(draft_lens.size());
    ChromosomeReduceData& reduce_data = chrom_reduce_data.at(0);
    reduce_data.seq_id = 0;
    reduce_data.seq_name = draft_lens.at(0).first;
    reduce_data.seq_len = draft_lens.at(0).second;
    reduce_data.num_bam_regions = static_cast<int64_t>(std::size(bam_windows));
    reduce_data.remaining_bam_regions = static_cast<int64_t>(std::size(bam_windows));

    // Create the input async queue and populate it.
    utils::AsyncQueue<secondary::Window> input_queue{8};
    for (const auto& bam_window : bam_windows) {
        CATCH_REQUIRE(input_queue.try_push(secondary::Window{bam_window}) ==
                      utils::AsyncQueueStatus::Success);
    }
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Create the output queue.
    utils::AsyncQueue<InferenceData> output_queue{256};

    // Workflow controls.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    // Tracker stats.
    secondary::Stats stats;
    stats.set("processed", 0.0);

    // Unit under test.
    worker_sample_producer(input_queue, output_queue, chrom_reduce_data, resources, stats,
                           worker_terminate, ret_status, bam_regions, draft_lens, draft_seqs,
                           secondary::VariantCandidateSource::COMPUTE, std::nullopt,
                           /*num_threads=*/2, /*window_len=*/1000, /*window_overlap=*/0,
                           /*variant_flanking_bases=*/50, /*continue_on_exception=*/false,
                           /*ploidy=*/2, /*pass_min_qual=*/3.0f, /*tiled_regions=*/false,
                           /*tiled_ext_flanks=*/false, /*tiled_ext_major=*/0,
                           /*tiled_ext_min_cov=*/0, /*tiled_ext_cov_fract=*/0.0f,
                           /*min_depth=*/0);

    // Eval.
    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());
    CATCH_CHECK(reduce_data.ready);
    CATCH_CHECK(reduce_data.remaining_bam_regions == 0);

    // clang-format off
    const std::vector<secondary::Variant> expected_variants{
            secondary::Variant{
                    .seq_id = 0,
                    .pos = 93,
                    .ref = "C",
                    .alts = {"T"},
                    .filter = "PASS",
                    .info = {},
                    .qual = 60.0f,
                    .genotype = {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}},
                    .rstart = 0,
                    .rend = 0,
            },
            secondary::Variant{
                    .seq_id = 0,
                    .pos = 1471,
                    .ref = "T",
                    .alts = {"G"},
                    .filter = "PASS",
                    .info = {},
                    .qual = 60.0f,
                    .genotype = {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}},
                    .rstart = 0,
                    .rend = 0,
            },
            secondary::Variant{
                    .seq_id = 0,
                    .pos = 7429,
                    .ref = "C",
                    .alts = {"T"},
                    .filter = "PASS",
                    .info = {},
                    .qual = 60.0f,
                    .genotype = {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "15"}},
                    .rstart = 0,
                    .rend = 0,
            },
    };
    // clang-format on

    // Compare variants after sorting.
    std::vector<secondary::Variant> actual_variants = reduce_data.variants_simple;
    std::sort(std::begin(actual_variants), std::end(actual_variants),
              [](const secondary::Variant& lhs, const secondary::Variant& rhs) {
                  return lhs.pos < rhs.pos;
              });
    CATCH_CHECK(actual_variants == expected_variants);

    // Check the output queue that the samples were added.
    int64_t queued_samples = 0;
    InferenceData item;
    utils::AsyncQueueStatus pop_status = utils::AsyncQueueStatus::Success;
    while ((pop_status = output_queue.try_pop(item)) != utils::AsyncQueueStatus::Terminate) {
        CATCH_REQUIRE(pop_status == utils::AsyncQueueStatus::Success);
        CATCH_REQUIRE(std::size(item.samples) == 1);
        CATCH_CHECK(item.samples.front().seq_id == 0);
        CATCH_CHECK(!std::empty(item.samples.front().positions_major));
        ++queued_samples;
    }
    CATCH_CHECK(queued_samples == reduce_data.num_samples);
}

CATCH_TEST_CASE("worker_variant_writer frees per-chromosome output buffers after writing",
                TEST_GROUP) {
    /**
     * \brief Verifies that worker_variant_writer flushes completed chromosome results to disk and
     *          then releases the corresponding in-memory buffers.
     *
     *          The test seeds merged, simple, and inference variants plus processed regions as if
     *          reduction had already completed. The expected result is that all output files become
     *          non-empty and the per-chromosome vectors in chrom_reduce_data are cleared and
     *          swapped down to zero capacity.
     */

    const auto temp_dir = make_temp_dir("variant_writer");
    const auto merged_vcf = temp_dir.m_path / "merged.vcf";
    const auto simple_vcf = temp_dir.m_path / "simple.vcf";
    const auto inference_vcf = temp_dir.m_path / "inference.vcf";
    const auto processed_regions_bed = temp_dir.m_path / "processed_regions.bed";

    // VCF filters.
    const std::vector<std::pair<std::string, std::string>> filters{};

    // Input reference sequences.
    const std::vector<std::pair<std::string, int64_t>> contigs = {
            {"chr1", 100},
    };

    // Input variants to write.
    const secondary::Variant merged_variant{
            .seq_id = 0,
            .pos = 10,
            .ref = "A",
            .alts = {"T"},
            .filter = "PASS",
            .info = {},
            .qual = 60.0f,
            .genotype = {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "1"}},
            .rstart = 0,
            .rend = 0,
    };
    const secondary::Variant simple_variant{
            .seq_id = 0,
            .pos = 20,
            .ref = "C",
            .alts = {"G"},
            .filter = "PASS",
            .info = {},
            .qual = 50.0f,
            .genotype = {{"GT", "0/1"}, {"GQ", "50"}, {"DP", "."}},
            .rstart = 0,
            .rend = 0,
    };
    const secondary::Variant inference_variant{
            .seq_id = 0,
            .pos = 30,
            .ref = "G",
            .alts = {"A"},
            .filter = "PASS",
            .info = {},
            .qual = 40.0f,
            .genotype = {{"GT", "1/1"}, {"GQ", "40"}, {"DP", "10"}},
            .rstart = 0,
            .rend = 0,
    };

    // Initialize the chrom_reduce_data as it would be after all processing for
    // this chromosome is done.
    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    chrom_reduce_data[0].seq_id = 0;
    chrom_reduce_data[0].seq_name = "chr1";
    chrom_reduce_data[0].seq_len = 100;
    chrom_reduce_data[0].variants_merged = {merged_variant};
    chrom_reduce_data[0].variants_simple = {simple_variant};
    chrom_reduce_data[0].variants_inference = {inference_variant};
    chrom_reduce_data[0].processed_regions = {
            secondary::IntervalInt64{5, 35, 0},
    };

    chrom_reduce_data[0].variants_merged.reserve(16);
    chrom_reduce_data[0].variants_simple.reserve(16);
    chrom_reduce_data[0].variants_inference.reserve(16);
    chrom_reduce_data[0].processed_regions.reserve(16);

    // Populate the input queue.
    utils::AsyncQueue<int64_t> input_queue{2};
    CATCH_REQUIRE(input_queue.try_push(0) == utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Workflow control parameters.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    // Run worker_variant_writer.
    {
        secondary::VCFWriter merged_writer(merged_vcf, filters, contigs, false);
        std::optional<secondary::VCFWriter> simple_writer{std::in_place, simple_vcf, filters,
                                                          contigs, false};
        std::optional<secondary::VCFWriter> inference_writer{std::in_place, inference_vcf, filters,
                                                             contigs, false};
        std::ofstream ofs_regions(processed_regions_bed);

        worker_variant_writer(input_queue, chrom_reduce_data, worker_terminate, ret_status,
                              merged_writer, ofs_regions, simple_writer, inference_writer, false);
    }

    // Check that no exceptions were thrown.
    CATCH_CHECK(!ret_status.exception_thrown);
    CATCH_CHECK(!worker_terminate.load());

    // Check that the chrom_reduce_data buffers were cleared for this chromosome.
    CATCH_CHECK(chrom_reduce_data[0].variants_merged.empty());
    CATCH_CHECK(chrom_reduce_data[0].variants_merged.capacity() == 0);
    CATCH_CHECK(chrom_reduce_data[0].variants_simple.empty());
    CATCH_CHECK(chrom_reduce_data[0].variants_simple.capacity() == 0);
    CATCH_CHECK(chrom_reduce_data[0].variants_inference.empty());
    CATCH_CHECK(chrom_reduce_data[0].variants_inference.capacity() == 0);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions.empty());
    CATCH_CHECK(chrom_reduce_data[0].processed_regions.capacity() == 0);

    // Check that the writer wrote data on disk.
    CATCH_CHECK(std::filesystem::file_size(merged_vcf) > 0);
    CATCH_CHECK(std::filesystem::file_size(simple_vcf) > 0);
    CATCH_CHECK(std::filesystem::file_size(inference_vcf) > 0);
    CATCH_CHECK(std::filesystem::file_size(processed_regions_bed) > 0);
}

CATCH_TEST_CASE(
        "worker_variant_calling_reduce keeps simple variants when flank trimming collapses a "
        "window",
        TEST_GROUP) {
    /**
     * \brief Verifies that worker_variant_calling_reduce keeps simple variants when inference for
     *          a window collapses away after trimming.
     *
     *          The input sample only spans positions that trim down to an empty callable region, so
     *          no inference variants should survive. The expected result is that the reducer still
     *          records the processed interval, emits the chromosome as ready, and preserves the
     *          existing simple variant in variants_merged.
     */
    const auto temp_dir = make_temp_dir("variant_reduce_trim");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nAAAAAA\n";
    }

    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);
    secondary::VariantCallingSample vc_sample{
            .seq_id = 0,
            .positions_major = {4, 5},
            .positions_minor = {0, 0},
            .logits = make_polyploid_probs(decoder.get_label_scheme_symbols(), {"AA", "AA"},
                                           {0.999f, 0.999f}),
            .depth = at::ones({2}, at::kInt),
    };
    const secondary::Variant simple_variant{
            .seq_id = 0,
            .pos = 4,
            .ref = "A",
            .alts = {"T"},
            .filter = "PASS",
            .info = {},
            .qual = 60.0f,
            .genotype = {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "."}},
            .rstart = 0,
            .rend = 0,
    };

    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 6;
        data.progress_target = 2;
        data.ready = true;
        data.num_samples = 1;
        data.variants_simple = {simple_variant};
    }

    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 6},
    };

    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    utils::AsyncQueue<int64_t> output_queue{2};
    CATCH_REQUIRE(input_queue.try_push(std::move(vc_sample)) == utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    secondary::Stats stats;
    stats.set("processed", 0.0);

    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  {}, 30.0f, false, false, 2, 1,
                                  secondary::VariantCandidateSource::COMPUTE);

    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());
    CATCH_REQUIRE(std::empty(chrom_reduce_data[0].variants_inference));
    CATCH_CHECK(chrom_reduce_data[0].variants_merged == std::vector{simple_variant});
    CATCH_REQUIRE(std::size(chrom_reduce_data[0].processed_regions) == 1);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].start == 4);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].stop == 5);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].value == 0);
    CATCH_CHECK(std::size(output_queue) == 1);
    CATCH_CHECK(stats.get_stats().at("processed") == 2.0);

    int64_t ready_seq_id = -1;
    CATCH_REQUIRE(output_queue.try_pop(ready_seq_id) == utils::AsyncQueueStatus::Success);
    CATCH_CHECK(ready_seq_id == 0);
}

CATCH_TEST_CASE(
        "worker_variant_calling_reduce trims inference variants before right trimmed flank simple "
        "variants",
        TEST_GROUP) {
    /*
     * This reproduces the right-flank leak that happens if an inference variant can extend
     * beyond the callable merge region and into the right trimmed flank.
     *
     *                  0    5    10  13
     *                  |----|----|--|
     *   reference      AAAAATTTCCCGG
     *   processed      [============)  [0,13)
     *   callable            [==)       [5,8) after flank_trim=5
     *   Inference           ACCGGG     [5,11) TTTCCC -> ACCGGG
     *   Inf-final           AC         [5,7)  TTT -> ACC           -> First variant
     *   Kadayashi             CGGG     positions 7,8,9,10          -> Second variant
     *
     * Decoding is clipped to the callable merge region so the retained inference variant ends
     * before the right-flank simple SNPs.
     */
    const auto temp_dir = make_temp_dir("variant_reduce_right_trim_overlap");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nAAAAATTTCCCGG\n";
    }

    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);
    secondary::VariantCallingSample vc_sample{
            .seq_id = 0,
            .positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            .positions_minor = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            .logits = make_polyploid_probs(decoder.get_label_scheme_symbols(),
                                           {"AAAAAACCGGGGG", "AAAAAACCGGGGG"},
                                           {0.999999f, 0.999999f}),
            .depth = at::ones({13}, at::kInt)};

    const auto make_simple_snp = [](const int64_t pos, const std::string& ref,
                                    const std::string& alt) {
        return secondary::Variant{
                .seq_id = 0,
                .pos = pos,
                .ref = ref,
                .alts = {alt},
                .filter = "PASS",
                .info = {},
                .qual = 60.0f,
                .genotype = {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "1"}},
                .rstart = 0,
                .rend = 0,
        };
    };
    const std::vector<secondary::Variant> simple_variants{
            make_simple_snp(7, "T", "C"),
            make_simple_snp(8, "C", "G"),
            make_simple_snp(9, "C", "G"),
            make_simple_snp(10, "C", "G"),
    };

    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 13;
        data.progress_target = 13;
        data.ready = true;
        data.num_samples = 1;
        data.variants_simple = simple_variants;
    }

    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 13},
    };

    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    utils::AsyncQueue<int64_t> output_queue{2};
    CATCH_REQUIRE(input_queue.try_push(std::move(vc_sample)) == utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    secondary::Stats stats;
    stats.set("processed", 0.0);

    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  {}, 30.0f, false, false, 2, 5,
                                  secondary::VariantCandidateSource::COMPUTE);

    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());

    const std::vector<secondary::Variant> expected_inference_variants{
            secondary::Variant{0,
                               5,
                               "TT",
                               {"AC"},
                               "PASS",
                               {},
                               53.922f,
                               {{"GT", "1/1"}, {"GQ", "54"}, {"DP", "1"}},
                               0,
                               2},
    };

    std::vector<secondary::Variant> expected_merged_variants = expected_inference_variants;
    expected_merged_variants.insert(std::end(expected_merged_variants), std::begin(simple_variants),
                                    std::end(simple_variants));

    CATCH_CHECK(chrom_reduce_data[0].variants_inference == expected_inference_variants);
    CATCH_CHECK(chrom_reduce_data[0].variants_merged == expected_merged_variants);

    CATCH_REQUIRE(std::size(chrom_reduce_data[0].processed_regions) == 1);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].start == 0);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].stop == 12);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].value == 0);
    CATCH_CHECK(std::size(output_queue) == 1);
    CATCH_CHECK(stats.get_stats().at("processed") == 13.0);
}

CATCH_TEST_CASE(
        "worker_variant_calling_reduce trims inference variants after left trimmed flank simple "
        "variants",
        TEST_GROUP) {
    /*
     * This reproduces the left-flank leak that happens if a long simple/Kadayashi variant starts
     * before the callable merge region but extends into a retained inference variant.
     *
     *                  0    5    10   16
     *                  |----|----|----|
     *   reference      AAAAATTTCCCGGGAA
     *   processed      [===============)   [0,16)
     *   callable            [=====)        [5,11) after flank_trim=5
     *   Kadayashi        CCCCCC            [2,8)  AAATTT -> CCCCCC       -> First variant
     *   Inference           ACCGGG         [5,11) TTTCCC -> ACCGGG
     *   Inf-final              GGG         [8,11) CCC -> GGG             -> Second variant
     *
     * Decoding starts after the longest left-overlapping simple variant so the simple variant
     * is kept and the retained inference variant starts after it.
     */
    const auto temp_dir = make_temp_dir("variant_reduce_left_trim_overlap");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nAAAAATTTCCCGGGAA\n";
    }

    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);
    secondary::VariantCallingSample vc_sample{
            .seq_id = 0,
            .positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            .positions_minor = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            .logits = make_polyploid_probs(decoder.get_label_scheme_symbols(),
                                           {"AAAAAACCGGGGGGAA", "AAAAAACCGGGGGGAA"},
                                           {0.999999f, 0.999999f}),
            .depth = at::arange(16, at::kInt)};
    const secondary::Variant long_simple_variant{
            .seq_id = 0,
            .pos = 2,
            .ref = "AAATTT",
            .alts = {"CCCCCC"},
            .filter = "PASS",
            .info = {},
            .qual = 60.0f,
            .genotype = {{"GT", "1/1"}, {"GQ", "60"}},
            .rstart = 0,
            .rend = 0,
    };

    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 16;
        data.progress_target = 16;
        data.ready = true;
        data.num_samples = 1;
        data.variants_simple = {long_simple_variant};
    }

    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 16},
    };

    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    utils::AsyncQueue<int64_t> output_queue{2};
    CATCH_REQUIRE(input_queue.try_push(std::move(vc_sample)) == utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    secondary::Stats stats;
    stats.set("processed", 0.0);

    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  {}, 30.0f, false, false, 2, 5,
                                  secondary::VariantCandidateSource::COMPUTE);

    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());

    const std::vector<secondary::Variant> expected_inference_variants{
            secondary::Variant{0,
                               8,
                               "CCC",
                               {"GGG"},
                               "PASS",
                               {},
                               52.161f,
                               {{"GT", "1/1"}, {"GQ", "52"}, {"DP", "9"}, {"MIN_DP", "8"}},
                               0,
                               3},
    };
    const std::vector<secondary::Variant> expected_merged_variants{
            long_simple_variant, expected_inference_variants.front()};
    CATCH_CHECK(chrom_reduce_data[0].variants_inference == expected_inference_variants);
    CATCH_CHECK(chrom_reduce_data[0].variants_merged == expected_merged_variants);
    CATCH_REQUIRE(std::size(chrom_reduce_data[0].processed_regions) == 1);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].start == 0);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].stop == 15);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].value == 0);
    CATCH_CHECK(std::size(output_queue) == 1);
    CATCH_CHECK(stats.get_stats().at("processed") == 16.0);
}

CATCH_TEST_CASE(
        "worker_variant_calling_reduce caps incremental progress to the per-sequence target",
        TEST_GROUP) {
    /**
     * \brief Verifies that worker_variant_calling_reduce never increments progress beyond the
     *          configured per-sequence target.
     *
     *          The test deliberately sets progress_target lower than the raw processed-base
     *          contribution of the input sample. The expected result is a successful reduction, a
     *          ready output item for the chromosome, and a final "processed" stat capped at
     *          progress_target rather than the larger uncapped value.
     */
    const auto temp_dir = make_temp_dir("variant_reduce_cap");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    // Write a simple reference FASTA file which will be opened by a FastxRandomReader.
    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nAAAAAA\n";
    }

    // Define the input vc_sample for reduction.
    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);
    secondary::VariantCallingSample vc_sample{
            .seq_id = 0,
            .positions_major = {2, 3},
            .positions_minor = {0, 0},
            .logits = make_polyploid_probs(decoder.get_label_scheme_symbols(), {"AA", "AA"},
                                           {0.999f, 0.999f}),
            .depth = at::ones({2}, at::kInt),
    };

    // Define the chrom_reduce_data and mark it as ready for reduction with only one sample to expect.
    // clang-format off
    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 6;
        data.progress_target = 1;   // <- The progress_target will be smaller than 2x the sum of processed bases, and will be capped.
        data.ready = true;
        data.num_samples = 1;
    }
    // clang-format on

    // Define the FastxRandomReader objects needed for worker_variant_calling_reduce.
    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    // Input reference lengths.
    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 6},
    };

    // Populate the input queue.
    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    CATCH_REQUIRE(input_queue.try_push(std::move(vc_sample)) == utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Create the output queue.
    utils::AsyncQueue<int64_t> output_queue{2};

    // Workflow control parameters.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    // Tracker statistics.
    secondary::Stats stats;
    stats.set("processed", 0.0);

    // Run the unit under test.
    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  {}, 30.0f, false, false, 2, 1,
                                  secondary::VariantCandidateSource::COMPUTE);

    // Eval that the run was successful, that the stats processed was capped to 1.0
    // that the output is populated now.
    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());
    CATCH_CHECK(stats.get_stats().at("processed") == 1.0);
    CATCH_CHECK(std::size(output_queue) == 1);
}

CATCH_TEST_CASE("worker_variant_calling_reduce tops up progress when no inference samples arrive",
                TEST_GROUP) {
    /**
     * \brief Verifies that worker_variant_calling_reduce can finalize a ready chromosome even
     *          when it receives zero inference samples.
     *
     *          This covers chromosomes for which the upstream pipeline produced no decode work.
     *          The expected result is that the reducer still marks the chromosome complete, emits
     *          its seq_id to the output queue, and tops the progress counter up to
     *          progress_target.
     */

    const auto temp_dir = make_temp_dir("variant_reduce_no_samples");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    // Write a simple reference FASTA file which will be opened by a FastxRandomReader.
    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nAAAAAAAAAA\n";
    }

    // Define the input vc_sample for reduction.
    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 10;
        data.progress_target = 6;
        data.ready = true;
        data.num_samples = 0;
    }

    // Decoder.
    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);

    // Define the FastxRandomReader objects needed for worker_variant_calling_reduce.
    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    // Input reference lengths.
    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 10},
    };

    // Empty input queue.
    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Create the output queue.
    utils::AsyncQueue<int64_t> output_queue{2};

    // Workflow control parameters.
    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    // Tracker statistics.
    secondary::Stats stats;
    stats.set("processed", 0.0);

    // Run the unit under test.
    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  {}, 30.0f, false, false, 2, 1,
                                  secondary::VariantCandidateSource::COMPUTE);

    // Eval that the run was successful.
    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());

    // Eval that the progress was incremented to max even though there were zero samples inferred.
    CATCH_CHECK(stats.get_stats().at("processed") == 6.0);
    CATCH_CHECK(std::size(output_queue) == 1);

    // Check that the sequence was marked as processed and ready.
    int64_t ready_seq_id = -1;
    CATCH_REQUIRE(output_queue.try_pop(ready_seq_id) == utils::AsyncQueueStatus::Success);
    CATCH_CHECK(ready_seq_id == 0);
}

CATCH_TEST_CASE(
        "worker_variant_calling_reduce collapses variants to haploid when hemizygous_regions is "
        "set",
        TEST_GROUP) {
    /**
     * \brief Verifies that worker_variant_calling_reduce collapses hom diploid variants to haploid when
     *          homozygous_regions is set.
     */
    const auto temp_dir = make_temp_dir("variant_reduce_trim");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nAAAAAA\n";
    }

    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);
    secondary::VariantCallingSample vc_sample{
            .seq_id = 0,
            .positions_major = {4, 5},
            .positions_minor = {0, 0},
            .logits = make_polyploid_probs(decoder.get_label_scheme_symbols(), {"AA", "AA"},
                                           {0.999f, 0.999f}),
            .depth = at::tensor({4, 3}),
    };
    const secondary::Variant simple_variant{
            .seq_id = 0,
            .pos = 4,
            .ref = "A",
            .alts = {"T"},
            .filter = "PASS",
            .info = {},
            .qual = 60.0f,
            .genotype = {{"GT", "1/1"}, {"GQ", "60"}},
            .rstart = 0,
            .rend = 0,
    };
    const secondary::Variant expected_simple_variant{
            .seq_id = 0,
            .pos = 4,
            .ref = "A",
            .alts = {"T"},
            .filter = "PASS",
            .info = {},
            .qual = 60.0f,
            .genotype = {{"GT", "1"}, {"GQ", "60"}},
            .rstart = 0,
            .rend = 0,
    };
    const std::vector<std::vector<secondary::Region>> hemizygous_regions = {{
            {"ref", 0, 10},
    }};

    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 6;
        data.progress_target = 2;
        data.ready = true;
        data.num_samples = 1;
        data.variants_simple = {simple_variant};
    }

    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 6},
    };

    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    utils::AsyncQueue<int64_t> output_queue{2};
    CATCH_REQUIRE(input_queue.try_push(std::move(vc_sample)) == utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    secondary::Stats stats;
    stats.set("processed", 0.0);

    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  hemizygous_regions, 30.0f, false, false, 2, 1,
                                  secondary::VariantCandidateSource::COMPUTE);

    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());
    CATCH_REQUIRE(std::empty(chrom_reduce_data[0].variants_inference));
    CATCH_CHECK(chrom_reduce_data[0].variants_merged == std::vector{expected_simple_variant});
    CATCH_REQUIRE(std::size(chrom_reduce_data[0].processed_regions) == 1);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].start == 4);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].stop == 5);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].value == 0);
    CATCH_CHECK(std::size(output_queue) == 1);
    CATCH_CHECK(stats.get_stats().at("processed") == 2.0);

    int64_t ready_seq_id = -1;
    CATCH_REQUIRE(output_queue.try_pop(ready_seq_id) == utils::AsyncQueueStatus::Success);
    CATCH_CHECK(ready_seq_id == 0);
}

CATCH_TEST_CASE(
        "worker_variant_calling_reduce keeps variants in diploid region when hemizygous_regions is "
        "set",
        TEST_GROUP) {
    /**
     * \brief Verifies that worker_variant_calling_reduce keeps diploid variants in diploid regions when
     *          hemizygous_regions is set.
     */
    const auto temp_dir = make_temp_dir("variant_reduce_trim");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nAAAAAA\n";
    }

    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);
    secondary::VariantCallingSample vc_sample{
            .seq_id = 0,
            .positions_major = {4, 5},
            .positions_minor = {0, 0},
            .logits = make_polyploid_probs(decoder.get_label_scheme_symbols(), {"AA", "AA"},
                                           {0.999f, 0.999f}),
            .depth = at::tensor({4, 3}),
    };
    const secondary::Variant simple_variant{
            .seq_id = 0,
            .pos = 4,
            .ref = "A",
            .alts = {"T"},
            .filter = "PASS",
            .info = {},
            .qual = 60.0f,
            .genotype = {{"GT", "1/1"}, {"GQ", "60"}},
            .rstart = 0,
            .rend = 0,
    };
    const std::vector<std::vector<secondary::Region>> hemizygous_regions = {{{"ref", 6, 10}}};

    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 6;
        data.progress_target = 2;
        data.ready = true;
        data.num_samples = 1;
        data.variants_simple = {simple_variant};
    }

    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 6},
    };

    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    utils::AsyncQueue<int64_t> output_queue{2};
    CATCH_REQUIRE(input_queue.try_push(std::move(vc_sample)) == utils::AsyncQueueStatus::Success);
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);

    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    secondary::Stats stats;
    stats.set("processed", 0.0);

    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  hemizygous_regions, 30.0f, false, false, 2, 1,
                                  secondary::VariantCandidateSource::COMPUTE);

    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());
    CATCH_REQUIRE(std::empty(chrom_reduce_data[0].variants_inference));
    CATCH_CHECK(chrom_reduce_data[0].variants_merged == std::vector{simple_variant});
    CATCH_REQUIRE(std::size(chrom_reduce_data[0].processed_regions) == 1);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].start == 4);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].stop == 5);
    CATCH_CHECK(chrom_reduce_data[0].processed_regions[0].value == 0);
    CATCH_CHECK(std::size(output_queue) == 1);
    CATCH_CHECK(stats.get_stats().at("processed") == 2.0);

    int64_t ready_seq_id = -1;
    CATCH_REQUIRE(output_queue.try_pop(ready_seq_id) == utils::AsyncQueueStatus::Success);
    CATCH_CHECK(ready_seq_id == 0);
}

CATCH_TEST_CASE("filter_hemizygous_variants", TEST_GROUP) {
    /**
     * \brief Test handling of variant and hemizygous region overlaps in filter_hemizygous_variants.
     */

    // clang-format off
    const std::vector<std::vector<secondary::Region>> hemizygous_regions = {
        {{"chr1", 0, -1}},                    // entire contig region with end unspecified
        {},                                   // no regions specified
        {{"chr3", 10, 20}, {"chr3", 30, 40}}, // partial regions
    };

    const std::vector<secondary::Variant> chr1_variants = {
        {
            // collapsed because hom in hemizygous region
            .seq_id = 0, .pos = 0, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // filtered because het-alt
            .seq_id = 0, .pos = 5, .ref = "A", .alts = {"T", "C"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/2"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // filtered because het
            .seq_id = 0, .pos = 5, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "0/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }
    };
    const std::vector<secondary::Variant> expected_chr1_filtered_variants = {
        {
            // collapsed because hom in hemizygous region
            .seq_id = 0, .pos = 0, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }
    };
    // clang-format on

    const std::vector<secondary::Variant> chr1_filtered_variants =
            smallvar::filter_hemizygous_variants(chr1_variants, hemizygous_regions[0]);
    CATCH_CHECK(chr1_filtered_variants == expected_chr1_filtered_variants);

    // clang-format off
    const std::vector<secondary::Variant> chr2_variants = {
        {
            .seq_id = 1, .pos = 0, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            .seq_id = 1, .pos = 5, .ref = "A", .alts = {"T", "C"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/2"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            .seq_id = 1, .pos = 10, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "0/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }
    };
    // clang-format on

    const std::vector<secondary::Variant> chr2_filtered_variants =
            smallvar::filter_hemizygous_variants(chr2_variants, hemizygous_regions[1]);
    // all unchanged because hemizygous region list is empty
    CATCH_CHECK(chr2_filtered_variants == chr2_variants);

    // clang-format off
    const std::vector<secondary::Variant> chr3_variants = {
        {
            // unchanged because outside hemizygous region
            .seq_id = 2, .pos = 1, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because spans start of a hemizygous region
            .seq_id = 2, .pos = 8, .ref = "ACCGTGT", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // discarded because het in a hemizygous region
            .seq_id = 2, .pos = 16, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "0/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because spans end of a hemizygous region
            .seq_id = 2, .pos = 19, .ref = "AGAG", .alts = {"A"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "0/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because outside hemizygous region
            .seq_id = 2, .pos = 22, .ref = "A", .alts = {"C"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "0/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // collapsed because hom inside a hemizygous region
            .seq_id = 2, .pos = 33, .ref = "ACC", .alts = {"A"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because outside hemizygous region
            .seq_id = 2, .pos = 43, .ref = "G", .alts = {"GTTC"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        },
    };
    const std::vector<secondary::Variant> expected_chr3_filtered_variants = {
        {
            // unchanged because outside hemizygous region
            .seq_id = 2, .pos = 1, .ref = "A", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because spans start of a hemizygous region
            .seq_id = 2, .pos = 8, .ref = "ACCGTGT", .alts = {"T"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because spans end of a hemizygous region
            .seq_id = 2, .pos = 19, .ref = "AGAG", .alts = {"A"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "0/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because outside hemizygous region
            .seq_id = 2, .pos = 22, .ref = "A", .alts = {"C"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "0/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // collapsed because hom inside a hemizygous region
            .seq_id = 2, .pos = 33, .ref = "ACC", .alts = {"A"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        }, {
            // unchanged because outside hemizygous region
            .seq_id = 2, .pos = 43, .ref = "G", .alts = {"GTTC"}, .filter = "PASS", .info = {}, .qual = 60.0f, .genotype = {{"GT", "1/1"}, {"GQ", "60"}}, .rstart = 0, .rend = 0,
        },
    };
    // clang-format on

    const std::vector<secondary::Variant> chr3_filtered_variants =
            smallvar::filter_hemizygous_variants(chr3_variants, hemizygous_regions[2]);
    CATCH_CHECK(chr3_filtered_variants == expected_chr3_filtered_variants);
}

CATCH_TEST_CASE("worker_variant_calling_reduce emits diploid gVCF records around merged variants",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("variant_reduce_gvcf_no_samples");
    const auto reference_fasta = temp_dir.m_path / "reference.fa";

    {
        std::ofstream ref_out(reference_fasta);
        ref_out << ">chr1\nACGTG\n";
    }

    // clang-format off
    std::vector<ChromosomeReduceData> chrom_reduce_data(1);
    {
        ChromosomeReduceData& data = chrom_reduce_data[0];
        data.seq_id = 0;
        data.seq_name = "chr1";
        data.seq_len = 5;
        data.progress_target = 5;
        data.ready = true;
        data.num_samples = 0;
        data.selected_regions = {secondary::RegionInt{0, 0, 5}};
        data.variants_simple = {secondary::Variant{0, 1, "C", {"A"}, "PASS", {}, 41.0f,
                                                   {{"GT", "1/1"}, {"GQ", "41"}}, 0, 0},
                                secondary::Variant{0, 2, "G", {"T"}, "PASS", {}, 42.0f,
                                                   {{"GT", "1/1"}, {"GQ", "42"}}, 0, 0}};
        data.read_endpoints = {
            {0, 1},
            {0, 2},
            {0, 3},
            {0, 5},
            {1, 2},
            {1, 3},
            {1, 5},
            {2, 3},
            {2, 5},
            {3, 4}
        };
    }
    // clang-format on

    const secondary::DecoderBase decoder(secondary::LabelSchemeType::DIPLOID);

    std::vector<std::unique_ptr<hts_io::FastxRandomReader>> fastx_readers;
    fastx_readers.emplace_back(std::make_unique<hts_io::FastxRandomReader>(reference_fasta));

    const std::vector<std::pair<std::string, int64_t>> draft_lens{
            {"chr1", 5},
    };

    utils::AsyncQueue<secondary::VariantCallingSample> input_queue{2};
    input_queue.terminate(utils::AsyncQueueTerminateFast::No);
    utils::AsyncQueue<int64_t> output_queue{2};

    std::atomic<bool> worker_terminate{false};
    secondary::WorkerReturnStatus ret_status;

    secondary::Stats stats;
    stats.set("processed", 0.0);

    worker_variant_calling_reduce(input_queue, output_queue, chrom_reduce_data, worker_terminate,
                                  ret_status, stats, fastx_readers, false, 1, draft_lens, decoder,
                                  {}, 30.0f, false, true, 2, 1,
                                  secondary::VariantCandidateSource::COMPUTE);

    // clang-format off
    CATCH_REQUIRE(!ret_status.exception_thrown);
    CATCH_REQUIRE(!worker_terminate.load());
    CATCH_REQUIRE(chrom_reduce_data[0].variants_merged ==
                  std::vector{secondary::Variant{0, 0, "A", {"<*>"}, ".",
                                                 {{"END", "1"}}, 60.0f,
                                                 {{"GT", "0/0"}, {"GQ", "60"}, {"LEN", "1"}, {"MIN_DP", "4"}}, 0,
                                                 1},
                              secondary::Variant{0, 1, "C", {"A"}, "PASS", {}, 41.0f,
                                                 {{"GT", "1/1"}, {"GQ", "41"}}, 0, 0},
                              secondary::Variant{0, 2, "G", {"T"}, "PASS", {}, 42.0f,
                                                 {{"GT", "1/1"}, {"GQ", "42"}}, 0, 0},
                              secondary::Variant{0, 3, "T", {"<*>"}, ".",
                                                 {{"END", "5"}}, 60.0f,
                                                 {{"GT", "0/0"}, {"GQ", "60"}, {"LEN", "2"}, {"MIN_DP", "3"}}, 3,
                                                 5}});
    CATCH_CHECK(stats.get_stats().at("processed") == 5.0);
    CATCH_CHECK(std::size(output_queue) == 1);
    // clang-format on
}

}  // namespace dorado::smallvar::tests
