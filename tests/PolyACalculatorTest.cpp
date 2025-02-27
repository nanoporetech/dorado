#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "poly_tail/poly_tail_calculator_selector.h"
#include "poly_tail/poly_tail_config.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/PolyACalculatorNode.h"
#include "utils/sequence_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#define TEST_GROUP "[poly_a_estimator]"

namespace fs = std::filesystem;

using namespace dorado;

struct TestCase {
    int estimated_bases = 0;
    std::string test_dir;
    bool is_rna;
};

CATCH_TEST_CASE("PolyACalculator: Test polyT tail estimation", TEST_GROUP) {
    auto [gt, data, is_rna] = GENERATE(
            TestCase{134, "poly_a/r9_rev_cdna", false}, TestCase{32, "poly_a/r10_fwd_cdna", false},
            TestCase{39, "poly_a/rna002", true}, TestCase{76, "poly_a/rna004", true});

    CATCH_CAPTURE(data);
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    pipeline_desc.add_node<PolyACalculatorNode>({sink}, 2, 1000);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    fs::path data_dir = fs::path(get_data_dir(data));
    auto seq_file = data_dir / "seq.txt";
    auto signal_file = data_dir / "signal.tensor";
    auto moves_file = data_dir / "moves.bin";
    auto read = std::make_unique<SimplexRead>();
    read->read_common.seq = ReadFileIntoString(seq_file.string());
    read->read_common.qstring = std::string(read->read_common.seq.length(), '~');
    read->read_common.moves = ReadFileIntoVector(moves_file.string());
    read->read_common.model_stride = 5;
    torch::load(read->read_common.raw_data, signal_file.string());
    read->read_common.read_id = "read_id";
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();
    read->read_common.client_info->contexts()
            .register_context<const dorado::poly_tail::PolyTailCalculatorSelector>(
                    std::make_shared<dorado::poly_tail::PolyTailCalculatorSelector>(
                            "", is_rna, false, 1.f, 0.f));

    // Push a Read type.
    pipeline->push_message(std::move(read));

    pipeline->terminate(DefaultFlushOptions());

    CATCH_CHECK(messages.size() == 1);

    auto out = std::get<SimplexReadPtr>(std::move(messages[0]));
    CATCH_CHECK(out->read_common.rna_poly_tail_length == gt);
}

CATCH_TEST_CASE("PolyACalculator: Test polyT tail estimation with custom config", TEST_GROUP) {
    auto config = (fs::path(get_data_dir("poly_a/configs")) / "polya.toml").string();

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    pipeline_desc.add_node<PolyACalculatorNode>({sink}, 2, 1000);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    fs::path data_dir = fs::path(get_data_dir("poly_a/r9_rev_cdna"));
    auto seq_file = data_dir / "seq.txt";
    auto signal_file = data_dir / "signal.tensor";
    auto moves_file = data_dir / "moves.bin";
    auto read = std::make_unique<SimplexRead>();
    read->read_common.seq = ReadFileIntoString(seq_file.string());
    read->read_common.qstring = std::string(read->read_common.seq.length(), '~');
    read->read_common.moves = ReadFileIntoVector(moves_file.string());
    read->read_common.model_stride = 5;
    torch::load(read->read_common.raw_data, signal_file.string());
    read->read_common.read_id = "read_id";
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();
    read->read_common.client_info->contexts()
            .register_context<const dorado::poly_tail::PolyTailCalculatorSelector>(
                    std::make_shared<dorado::poly_tail::PolyTailCalculatorSelector>(
                            config, false, false, 1.f, 0.f));

    // Push a Read type.
    pipeline->push_message(std::move(read));

    pipeline->terminate(DefaultFlushOptions());

    CATCH_CHECK(messages.size() == 1);

    auto out = std::get<SimplexReadPtr>(std::move(messages[0]));
    CATCH_CHECK(out->read_common.rna_poly_tail_length == -1);
}

CATCH_TEST_CASE("PolyTailConfig: Test parsing file", TEST_GROUP) {
    CATCH_SECTION("Check failure with non-existent file.") {
        const std::string missing_file = "foo_bar_baz";
        CATCH_CHECK_THROWS_WITH(dorado::poly_tail::prepare_configs(missing_file),
                                "PolyA config file doesn't exist at foo_bar_baz");
    }

    CATCH_SECTION("Only one primer is provided") {
        const auto fmt = R"delim(
            [anchors]
                front_primer = "ACTG"
        )delim";
        std::stringstream buffer(fmt);

        CATCH_CHECK_THROWS_WITH(dorado::poly_tail::prepare_configs(buffer),
                                "Both front_primer and rear_primer must be provided in the PolyA "
                                "configuration file.");
    }

    CATCH_SECTION("Only one plasmid flank is provided") {
        const auto fmt = R"delim(
            [anchors]
                plasmid_rear_flank = "ACTG"
        )delim";
        std::stringstream buffer(fmt);

        CATCH_CHECK_THROWS_WITH(
                dorado::poly_tail::prepare_configs(buffer),
                "Both plasmid_front_flank and plasmid_rear_flank must be provided in the "
                "PolyA configuration file.");
    }

    CATCH_SECTION("Parse all supported configs") {
        const auto fmt = R"delim(
            [anchors]
                plasmid_front_flank = "CGTA"
                plasmid_rear_flank = "ACTG"
                front_primer = "AAAAAA"
                rear_primer = "GGGGGG"

            [tail]
                tail_interrupt_length = 10
        )delim";
        std::stringstream buffer(fmt);

        auto configs = dorado::poly_tail::prepare_configs(buffer);
        CATCH_REQUIRE(configs.size() == 1);
        const auto& config = configs.front();
        CATCH_CHECK(config.front_primer == "AAAAAA");
        CATCH_CHECK(config.rc_front_primer == "TTTTTT");
        CATCH_CHECK(config.rear_primer == "GGGGGG");
        CATCH_CHECK(config.rc_rear_primer == "CCCCCC");
        CATCH_CHECK(config.plasmid_front_flank == "CGTA");
        CATCH_CHECK(config.rc_plasmid_front_flank == "TACG");
        CATCH_CHECK(config.plasmid_rear_flank == "ACTG");
        CATCH_CHECK(config.rc_plasmid_rear_flank == "CAGT");
        CATCH_CHECK(config.is_plasmid);  // Since the plasmid flanks were specified
        CATCH_CHECK(config.tail_interrupt_length == 10);
    }

    CATCH_SECTION("Override config missing id") {
        const auto fmt = R"delim(
            [[overrides]]
            [[overrides]]
            [[overrides]]
        )delim";
        std::stringstream buffer(fmt);

        CATCH_CHECK_THROWS_WITH(dorado::poly_tail::prepare_configs(buffer),
                                "Missing barcode_id in override poly tail configuration.");
    }

    CATCH_SECTION("Override config duplicate id") {
        const auto fmt = R"delim(
            [[overrides]]
                barcode_id = "duplicate"
            [[overrides]]
                barcode_id = "duplicate"
            [[overrides]]
                barcode_id = "duplicate"
        )delim";
        std::stringstream buffer(fmt);

        CATCH_CHECK_THROWS_WITH(dorado::poly_tail::prepare_configs(buffer),
                                "Duplicate barcode_id found in poly tail config file.");
    }

    CATCH_SECTION("Default config contains barcode id") {
        const auto fmt = R"delim(
            barcode_id = "error"
            [[overrides]]
                barcode_id = "barcode0"
            [[overrides]]
                barcode_id = "barcode1"
            [[overrides]]
                barcode_id = "barcode2"
        )delim";
        std::stringstream buffer(fmt);

        CATCH_CHECK_THROWS_WITH(dorado::poly_tail::prepare_configs(buffer),
                                "Default poly tail config must not specify barcode_id.");
    }

    CATCH_SECTION("Parse override configs") {
        const int NUM_CONFIGS = 3;
        const auto fmt = R"delim(
            [tail]
                tail_interrupt_length = 10
            [[overrides]]
                barcode_id = "barcode0"
            [[overrides]]
                barcode_id = "barcode1"
            [[overrides]]
                barcode_id = "barcode2"
        )delim";
        std::stringstream buffer(fmt);

        auto configs = dorado::poly_tail::prepare_configs(buffer);
        CATCH_REQUIRE(configs.size() == NUM_CONFIGS + 1);  // specified configs + default
        for (int i = 0; i < NUM_CONFIGS; ++i) {
            CATCH_CHECK(configs[i].barcode_id ==
                        "barcode" + std::to_string(i));  // overridden value per config
            CATCH_CHECK(configs[i].tail_interrupt_length ==
                        10);  // default inherited from main config
        }
        CATCH_CHECK(configs.back().barcode_id.empty());
    }
}
