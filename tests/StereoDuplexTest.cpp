#include "TestUtils.h"
#include "read_pipeline/base/DefaultClientInfo.h"
#include "read_pipeline/base/messages/DuplexRead.h"
#include "read_pipeline/base/messages/ReadPair.h"
#include "read_pipeline/nodes/StereoDuplexEncoderNode.h"
#include "utils/sequence_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/serialize.h>

#include <algorithm>
#include <filesystem>

#define TEST_GROUP "[StereoDuplexTest]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(TEST_GROUP " " name, TEST_GROUP)

namespace {

std::filesystem::path DataPath(std::string_view filename) {
    return std::filesystem::path(get_stereo_data_dir()) / filename;
}

// materialise_read_raw_data takes a Message reference for the convenience of generic use.
// Here we have DuplexReadPtr, so this converts back and forth to get raw_data in the given
// DuplexReadPtr.
void generate_raw_data(dorado::DuplexReadPtr& duplex_read_ptr) {
    dorado::Message message = std::move(duplex_read_ptr);
    materialise_read_raw_data(message);
    duplex_read_ptr = message.take<dorado::DuplexReadPtr>();
}

}  // namespace

// Tests stereo encoder output for a real sample signal against known good output.
DEFINE_TEST("Encoder") {
    dorado::ReadPair read_pair;
    {
        auto& template_common = read_pair.template_read.read_common;
        template_common.seq = ReadFileIntoString(DataPath("template_seq"));
        template_common.qstring = ReadFileIntoString(DataPath("template_qstring"));
        template_common.moves = ReadFileIntoVector(DataPath("template_moves"));
        torch::load(template_common.raw_data, DataPath("template_raw_data.tensor").string());
        template_common.raw_data = template_common.raw_data.to(torch::kFloat16);
        template_common.run_id = "test_run";
        template_common.start_time_ms = 0;
        read_pair.template_read.seq_start = 0;
        read_pair.template_read.seq_end = template_common.seq.length();
    }
    {
        auto& complement_common = read_pair.complement_read.read_common;
        complement_common.seq = ReadFileIntoString(DataPath("complement_seq"));
        complement_common.qstring = ReadFileIntoString(DataPath("complement_qstring"));
        complement_common.moves = ReadFileIntoVector(DataPath("complement_moves"));
        torch::load(complement_common.raw_data, DataPath("complement_raw_data.tensor").string());
        complement_common.raw_data = complement_common.raw_data.to(torch::kFloat16);
        complement_common.start_time_ms = 100;
        read_pair.complement_read.seq_start = 0;
        read_pair.complement_read.seq_end = complement_common.seq.length();
    }

    at::Tensor stereo_raw_data;
    torch::load(stereo_raw_data, DataPath("stereo_raw_data.tensor").string());
    stereo_raw_data = stereo_raw_data.to(torch::kFloat16);

    const int input_signal_stride = 5;

    auto stereo_read = dorado::stereo_encode(read_pair, input_signal_stride);
    generate_raw_data(stereo_read);
    CATCH_REQUIRE(torch::equal(stereo_raw_data, stereo_read->read_common.raw_data));

    // Check that the duplex tag and run id is set correctly.
    CATCH_REQUIRE(stereo_read->read_common.is_duplex);
    CATCH_REQUIRE(stereo_read->read_common.run_id == read_pair.template_read.read_common.run_id);

    // Encode with swapped template and complement reads
    std::swap(read_pair.template_read, read_pair.complement_read);
    std::swap(read_pair.template_read.read_common.start_time_ms,
              read_pair.complement_read.read_common.start_time_ms);

    auto swapped_stereo_read = dorado::stereo_encode(read_pair, input_signal_stride);
    generate_raw_data(swapped_stereo_read);

    // Check if the encoded signal is NOT equal to the expected stereo_raw_data
    CATCH_REQUIRE(!torch::equal(stereo_raw_data, swapped_stereo_read->read_common.raw_data));
}

DEFINE_TEST("duplex read has correct fields set") {
    constexpr int input_signal_stride = 1;
    auto fill_common = [&](dorado::ReadPair::ReadData& read_data,
                           std::shared_ptr<dorado::ClientInfo> client_info, bool complement) {
        // Per-read values.
        read_data.read_common.num_minknow_events = complement ? 2 : 1;
        read_data.read_common.read_id = complement ? "complement_read" : "template_read";
        read_data.read_common.read_tag = complement ? 628 : 314;
        read_data.read_common.start_time_ms = complement ? 456 : 123;
        read_data.read_common.moves =
                complement ? std::vector<uint8_t>{1, 0, 0, 1} : std::vector<uint8_t>{1, 0, 1, 0};

        // Signal and sequence.
        constexpr int num_bases = 8;
        read_data.read_common.seq = "ATCGATCG";
        read_data.read_common.qstring = "12345678";
        if (complement) {
            dorado::utils::reverse_complement(read_data.read_common.seq);
            std::reverse(read_data.read_common.qstring.begin(),
                         read_data.read_common.qstring.end());
        }
        read_data.read_common.raw_data =
                at::rand({num_bases * input_signal_stride}, torch::kFloat16);
        read_data.seq_start = complement ? 2 : 1;
        read_data.seq_end = read_data.seq_start + 4;
        CATCH_REQUIRE(read_data.seq_end < num_bases);

        // Common values.
        read_data.read_common.acquisition_id = "test_acq";
        read_data.read_common.attributes.channel_number = 6;
        read_data.read_common.attributes.end_reason = "ended";
        read_data.read_common.attributes.filename = "file.pod5";
        read_data.read_common.attributes.is_end_reason_mux_change = false;
        read_data.read_common.attributes.model_stride = input_signal_stride;
        read_data.read_common.attributes.mux = 7;
        read_data.read_common.attributes.pore_type = "test_pore";
        read_data.read_common.attributes.read_number = 7;
        read_data.read_common.attributes.sample_rate = 500;
        read_data.read_common.attributes.start_time = "now";
        read_data.read_common.chemistry = dorado::models::Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ;
        read_data.read_common.client_info = std::move(client_info);
        read_data.read_common.experiment_id = "test_exp";
        read_data.read_common.flow_cell_product_code = "test_flowcell";
        read_data.read_common.flowcell_id = "fcid";
        read_data.read_common.is_duplex = false;
        read_data.read_common.mean_qscore_start_pos = 1;
        read_data.read_common.model_name = "fast";
        read_data.read_common.num_minknow_events = 2;
        read_data.read_common.num_trimmed_samples = 3;
        read_data.read_common.parent_read_id = "";
        read_data.read_common.position_id = "sequencer_position";
        read_data.read_common.protocol_start_time_ms = 42;
        read_data.read_common.rapid_chemistry = dorado::models::RapidChemistry::NONE;
        read_data.read_common.run_id = "test_run";
        read_data.read_common.sample_id = "test_sample";
        read_data.read_common.scale = 1.23f;
        read_data.read_common.scaling_method = "scaled";
        read_data.read_common.sequencing_kit = "test_kit";
        read_data.read_common.shift = 4.56f;
        read_data.read_common.split_count = 2;
        read_data.read_common.split_point = 1;
        read_data.read_common.subread_id = 1;
    };

    // Make a read pair.
    dorado::ReadPair read_pair;
    const auto client_info = std::make_shared<dorado::DefaultClientInfo>();
    fill_common(read_pair.template_read, client_info, false);
    fill_common(read_pair.complement_read, client_info, true);

    // Convert to a stereo read.
    const auto stereo_read = dorado::stereo_encode(read_pair, input_signal_stride);
    const auto& stereo_common = stereo_read->read_common;

    // Check that the read fields match expected values.
    CATCH_CHECK(stereo_common.acquisition_id == "test_acq");
    CATCH_CHECK(stereo_common.attributes.channel_number == 6);
    CATCH_CHECK(stereo_common.attributes.end_reason == "unknown");  // not set
    CATCH_CHECK(stereo_common.attributes.filename == "file.pod5");
    CATCH_CHECK(stereo_common.attributes.is_end_reason_mux_change == false);  // not set
    CATCH_CHECK(stereo_common.attributes.model_stride == input_signal_stride);
    CATCH_CHECK(stereo_common.attributes.mux == 7);
    CATCH_CHECK(stereo_common.attributes.pore_type == "test_pore");
    CATCH_CHECK(stereo_common.attributes.read_number == -1);  // not set
    CATCH_CHECK(stereo_common.attributes.sample_rate == 500);
    CATCH_CHECK(stereo_common.attributes.start_time == "now");
    CATCH_CHECK(stereo_common.chemistry == dorado::models::Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ);
    CATCH_CHECK(stereo_common.client_info == client_info);
    CATCH_CHECK(stereo_common.experiment_id == "test_exp");
    CATCH_CHECK(stereo_common.flow_cell_product_code == "test_flowcell");
    CATCH_CHECK(stereo_common.flowcell_id == "fcid");
    CATCH_CHECK(stereo_common.is_duplex == true);
    CATCH_CHECK(stereo_common.mean_qscore_start_pos == 0);  // not set
    CATCH_CHECK(stereo_common.model_name == "fast");
    CATCH_CHECK(stereo_common.moves.empty());             // moved to features
    CATCH_CHECK(stereo_common.num_minknow_events == 0);   // not set
    CATCH_CHECK(stereo_common.num_trimmed_samples == 0);  // not set
    CATCH_CHECK(stereo_common.parent_read_id == "");      // not set
    CATCH_CHECK(stereo_common.position_id == "sequencer_position");
    CATCH_CHECK(stereo_common.protocol_start_time_ms == 42);
    CATCH_CHECK(stereo_common.qstring.empty());  // moved to features
    CATCH_CHECK(stereo_common.rapid_chemistry == dorado::models::RapidChemistry::NONE);
    CATCH_CHECK(stereo_common.raw_data.numel() == 0);  // moved to features
    CATCH_CHECK(stereo_common.read_id == "template_read;complement_read");
    CATCH_CHECK(stereo_common.read_tag == 314);
    CATCH_CHECK(stereo_common.run_id == "test_run");
    CATCH_CHECK(stereo_common.sample_id == "test_sample");
    CATCH_CHECK(stereo_common.scale == 0);            // not set
    CATCH_CHECK(stereo_common.scaling_method == "");  // not set
    CATCH_CHECK(stereo_common.seq.empty());           // moved to features
    CATCH_CHECK(stereo_common.sequencing_kit == "test_kit");
    CATCH_CHECK(stereo_common.shift == 0);        // not set
    CATCH_CHECK(stereo_common.split_count == 1);  // not set
    CATCH_CHECK(stereo_common.split_point == 0);  // not set
    CATCH_CHECK(stereo_common.start_time_ms == 123);
    CATCH_CHECK(stereo_common.subread_id == 0);  // not set

    // Check that the features are set correctly.
    const auto& features = stereo_read->stereo_feature_inputs;
    CATCH_CHECK(features.signal_stride == input_signal_stride);
    CATCH_CHECK(features.template_seq_start == 1);
    CATCH_CHECK(features.complement_seq_start == 2);
    CATCH_CHECK(features.template_seq == "ATCGATCG");
    CATCH_CHECK(features.complement_seq == "CGATCGAT");
    CATCH_CHECK(features.template_qstring == "12345678");
    CATCH_CHECK(features.complement_qstring == "87654321");
    CATCH_CHECK(features.template_moves == read_pair.template_read.read_common.moves);
    CATCH_CHECK(features.complement_moves == read_pair.complement_read.read_common.moves);
    CATCH_CHECK(at::equal(features.template_signal, read_pair.template_read.read_common.raw_data));
    CATCH_CHECK(at::equal(features.complement_signal,
                          read_pair.complement_read.read_common.raw_data.flip(0)));
}
