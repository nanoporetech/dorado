#include "TestUtils.h"
#include "read_pipeline/base/messages/DuplexRead.h"
#include "read_pipeline/base/messages/ReadPair.h"
#include "read_pipeline/nodes/StereoDuplexEncoderNode.h"

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
