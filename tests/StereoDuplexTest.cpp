#include "TestUtils.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/StereoDuplexEncoderNode.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <vector>

#define TEST_GROUP "StereoDuplexTest"

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
    duplex_read_ptr = std::move(std::get<dorado::DuplexReadPtr>(message));
}

}  // namespace

// Tests stereo encoder output for a real sample signal against known good output.
CATCH_TEST_CASE(TEST_GROUP "Encoder") {
    dorado::ReadPair::ReadData template_read{};
    {
        template_read.read_common.seq = ReadFileIntoString(DataPath("template_seq"));
        template_read.read_common.qstring = ReadFileIntoString(DataPath("template_qstring"));
        template_read.read_common.moves = ReadFileIntoVector(DataPath("template_moves"));
        torch::load(template_read.read_common.raw_data,
                    DataPath("template_raw_data.tensor").string());
        template_read.read_common.raw_data = template_read.read_common.raw_data.to(torch::kFloat16);
        template_read.read_common.run_id = "test_run";
        template_read.read_common.start_time_ms = static_cast<uint64_t>(0);
        template_read.seq_start = 0;
        template_read.seq_end = template_read.read_common.seq.length();
    }

    dorado::ReadPair::ReadData complement_read{};
    {
        complement_read.read_common.seq = ReadFileIntoString(DataPath("complement_seq"));
        complement_read.read_common.qstring = ReadFileIntoString(DataPath("complement_qstring"));
        complement_read.read_common.moves = ReadFileIntoVector(DataPath("complement_moves"));
        torch::load(complement_read.read_common.raw_data,
                    DataPath("complement_raw_data.tensor").string());
        complement_read.read_common.raw_data =
                complement_read.read_common.raw_data.to(torch::kFloat16);
        complement_read.read_common.start_time_ms = static_cast<uint64_t>(100);
        complement_read.seq_start = 0;
        complement_read.seq_end = complement_read.read_common.seq.length();
    }

    at::Tensor stereo_raw_data;
    torch::load(stereo_raw_data, DataPath("stereo_raw_data.tensor").string());
    stereo_raw_data = stereo_raw_data.to(torch::kFloat16);

    dorado::StereoDuplexEncoderNode stereo_node = dorado::StereoDuplexEncoderNode(5);

    dorado::ReadPair read_pair;
    read_pair.template_read = template_read;
    read_pair.complement_read = complement_read;
    auto stereo_read = stereo_node.stereo_encode(read_pair);
    generate_raw_data(stereo_read);
    CATCH_REQUIRE(torch::equal(stereo_raw_data, stereo_read->read_common.raw_data));

    // Check that the duplex tag and run id is set correctly.
    CATCH_REQUIRE(stereo_read->read_common.is_duplex);
    CATCH_REQUIRE(stereo_read->read_common.run_id == template_read.read_common.run_id);

    // Encode with swapped template and complement reads
    std::swap(read_pair.template_read, read_pair.complement_read);
    std::swap(read_pair.template_read.read_common.start_time_ms,
              read_pair.complement_read.read_common.start_time_ms);

    auto swapped_stereo_read = stereo_node.stereo_encode(read_pair);
    generate_raw_data(swapped_stereo_read);

    // Check if the encoded signal is NOT equal to the expected stereo_raw_data
    CATCH_REQUIRE(!torch::equal(stereo_raw_data, swapped_stereo_read->read_common.raw_data));
}
