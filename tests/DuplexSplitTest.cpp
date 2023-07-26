#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/DuplexSplitNode.h"
#include "read_pipeline/NullNode.h"
#include "read_pipeline/PairingNode.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/StereoDuplexEncoderNode.h"
#include "read_pipeline/SubreadTaggerNode.h"

#include <catch2/catch.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <vector>

#define TEST_GROUP "[DuplexSplitTest]"

namespace {
std::filesystem::path DataPath(std::string_view filename) {
    return std::filesystem::path(get_split_data_dir()) / filename;
}

std::shared_ptr<dorado::Read> make_read() {
    std::shared_ptr<dorado::Read> read = std::make_shared<dorado::Read>();
    read->range = 0;
    read->sample_rate = 4000;
    read->offset = -287;
    read->scaling = 0.14620706;
    read->shift = 94.717316;
    read->scale = 26.888939;
    read->model_stride = 5;
    read->read_id = "00a2dd45-f6a9-49ba-86ee-5d2a37b861cb";
    read->num_trimmed_samples = 10;
    read->attributes.read_number = 321;
    read->attributes.channel_number = 664;
    read->attributes.mux = 3;
    read->attributes.start_time = "2023-02-21T12:46:01.526+00:00";
    read->attributes.num_samples = 256790;
    read->start_sample = 29767426;
    read->end_sample = 30024216;
    read->run_acquisition_start_time_ms = 1676976119670;

    read->seq = ReadFileIntoString(DataPath("seq"));
    read->qstring = ReadFileIntoString(DataPath("qstring"));
    read->moves = ReadFileIntoVector(DataPath("moves"));
    torch::load(read->raw_data, DataPath("raw.tensor").string());
    read->raw_data = read->raw_data.to(torch::kFloat16);
    read->read_tag = 42;

    return read;
}
}  // namespace

TEST_CASE("4 subread splitting test", TEST_GROUP) {
    const auto read = make_read();

    dorado::DuplexSplitSettings splitter_settings;
    dorado::DuplexSplitNode splitter_node(splitter_settings, 1);

    const auto split_res = splitter_node.split(read);
    REQUIRE(split_res.size() == 4);
    std::vector<int> split_sizes;
    for (auto &r : split_res) {
        split_sizes.push_back(r->seq.size());
    }
    REQUIRE(split_sizes == std::vector<int>{6858, 7854, 5184, 5168});

    std::vector<std::string> start_times;
    for (auto &r : split_res) {
        start_times.push_back(r->attributes.start_time);
    }
    REQUIRE(start_times == std::vector<std::string>{"2023-02-21T12:46:01.529+00:00",
                                                    "2023-02-21T12:46:25.837+00:00",
                                                    "2023-02-21T12:46:39.607+00:00",
                                                    "2023-02-21T12:46:53.105+00:00"});

    std::vector<uint64_t> start_time_mss;
    for (auto &r : split_res) {
        start_time_mss.push_back(r->start_time_ms);
    }
    REQUIRE(start_time_mss ==
            std::vector<uint64_t>{1676983561529, 1676983585837, 1676983599607, 1676983613105});

    std::vector<uint64_t> num_sampless;
    for (auto &r : split_res) {
        num_sampless.push_back(r->attributes.num_samples);
    }
    REQUIRE(num_sampless == std::vector<uint64_t>{97125, 55055, 53940, 50475});

    std::set<std::string> names;
    for (auto &r : split_res) {
        names.insert(r->read_id);
    }
    REQUIRE(names.size() == 4);
    REQUIRE(std::all_of(split_res.begin(), split_res.end(),
                        [](const auto &r) { return r->read_tag == 42; }));
}

TEST_CASE("4 subread split tagging", TEST_GROUP) {
    const auto read = make_read();

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 3, messages);
    auto tag_node = pipeline_desc.add_node<dorado::SubreadTaggerNode>({sink});
    auto stereo_node =
            pipeline_desc.add_node<dorado::StereoDuplexEncoderNode>({tag_node}, read->model_stride);
    auto pairing_node = pipeline_desc.add_node<dorado::PairingNode>({stereo_node},
                                                                    dorado::ReadOrder::BY_CHANNEL);
    auto splitter_node = pipeline_desc.add_node<dorado::DuplexSplitNode>(
            {pairing_node}, dorado::DuplexSplitSettings{}, 1);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    pipeline->push_message(read);
    pipeline.reset();

    auto reads = ConvertMessages<std::shared_ptr<dorado::Read>>(messages);

    REQUIRE(reads.size() == 5);

    std::vector<size_t> expected_subread_ids = {0, 1, 2, 3, 4};
    std::vector<size_t> subread_ids;
    for (const auto &subread : reads) {
        subread_ids.push_back(subread->subread_id);
    }

    std::sort(std::begin(subread_ids), std::end(subread_ids));
    CHECK(subread_ids == expected_subread_ids);
    CHECK(std::all_of(reads.begin(), reads.end(),
                      [split_count = expected_subread_ids.size()](const auto &read) {
                          return read->split_count == split_count;
                      }));
    CHECK(std::all_of(reads.begin(), reads.end(), [](const auto &r) { return r->read_tag == 42; }));
}
