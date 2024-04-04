#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/PairingNode.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/ReadSplitNode.h"
#include "read_pipeline/StereoDuplexEncoderNode.h"
#include "read_pipeline/SubreadTaggerNode.h"
#include "splitter/DuplexReadSplitter.h"
#include "splitter/ReadSplitter.h"

#include <torch/torch.h>
// Catch2 must come after torch since both define CHECK()
#include <catch2/catch.hpp>

#include <algorithm>
#include <filesystem>
#include <vector>

#define TEST_GROUP "[DuplexSplitTest]"

namespace {
std::filesystem::path DataPath(std::string_view filename) {
    return std::filesystem::path(get_data_dir("split")) / filename;
}

auto make_read() {
    auto read = std::make_unique<dorado::SimplexRead>();
    read->range = 0;
    read->read_common.sample_rate = 4000;
    read->offset = -287;
    read->scaling = 0.14620706f;
    read->read_common.shift = 94.717316f;
    read->read_common.scale = 26.888939f;
    read->read_common.model_stride = 5;
    read->read_common.read_id = "00a2dd45-f6a9-49ba-86ee-5d2a37b861cb";
    read->read_common.num_trimmed_samples = 10;
    read->read_common.attributes.read_number = 321;
    read->read_common.attributes.channel_number = 664;
    read->read_common.attributes.mux = 3;
    read->read_common.attributes.start_time = "2023-02-21T12:46:01.526+00:00";
    read->read_common.attributes.num_samples = 256790;
    read->start_sample = 29767426;
    read->end_sample = 30024216;
    read->run_acquisition_start_time_ms = 1676976119670;

    read->read_common.seq = ReadFileIntoString(DataPath("seq"));
    read->read_common.qstring = ReadFileIntoString(DataPath("qstring"));
    read->read_common.moves = ReadFileIntoVector(DataPath("moves"));
    torch::load(read->read_common.raw_data, DataPath("raw.tensor").string());
    read->read_common.raw_data = read->read_common.raw_data.to(at::ScalarType::Half);
    read->read_common.read_tag = 42;
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();

    read->prev_read = "prev";
    read->next_read = "next";

    return read;
}
}  // namespace

TEST_CASE("4 subread splitting test", TEST_GROUP) {
    auto read = make_read();

    dorado::splitter::DuplexReadSplitter splitter_node(
            dorado::splitter::DuplexSplitSettings(false));

    const auto split_res = splitter_node.split(std::move(read));
    CHECK(split_res.size() == 4);
    std::vector<int> split_sizes;
    for (auto &r : split_res) {
        split_sizes.push_back(int(r->read_common.seq.size()));
    }
    CHECK(split_sizes == std::vector<int>{6858, 7854, 5185, 5168});

    std::vector<std::string> start_times;
    for (auto &r : split_res) {
        start_times.push_back(r->read_common.attributes.start_time);
    }
    CHECK(start_times == std::vector<std::string>{
                                 "2023-02-21T12:46:01.529+00:00", "2023-02-21T12:46:25.837+00:00",
                                 "2023-02-21T12:46:39.607+00:00", "2023-02-21T12:46:53.105+00:00"});

    std::vector<uint64_t> start_time_mss;
    for (auto &r : split_res) {
        start_time_mss.push_back(r->read_common.start_time_ms);
    }
    CHECK(start_time_mss ==
          std::vector<uint64_t>{1676983561529, 1676983585837, 1676983599607, 1676983613105});

    std::vector<uint64_t> num_samples;
    for (auto &r : split_res) {
        num_samples.push_back(r->read_common.attributes.num_samples);
    }
    CHECK(num_samples == std::vector<uint64_t>{97125, 55055, 53950, 50475});

    std::vector<uint32_t> split_points;
    for (auto &r : split_res) {
        split_points.push_back(r->read_common.split_point);
    }
    CHECK(split_points == std::vector<uint32_t>{0, 97230, 152310, 206305});

    std::set<std::string> names;
    for (auto &r : split_res) {
        names.insert(r->read_common.read_id);
    }
    CHECK(names.size() == 4);
    CHECK(std::all_of(split_res.begin(), split_res.end(),
                      [](const auto &r) { return r->read_common.read_tag == 42; }));

    std::vector<std::string> prev_read_names;
    std::vector<std::string> next_read_names;
    for (auto &r : split_res) {
        prev_read_names.push_back(r->prev_read);
        next_read_names.push_back(r->next_read);
    }
    CHECK(prev_read_names == std::vector<std::string>{"prev",
                                                      "e7e47439-5968-4883-96ff-7f2d2040dc43",
                                                      "a62e28ab-c367-4a93-af9b-84130d3df58c",
                                                      "f8e75422-3275-47f6-b45f-062aa00df368"});
    CHECK(next_read_names == std::vector<std::string>{"a62e28ab-c367-4a93-af9b-84130d3df58c",
                                                      "f8e75422-3275-47f6-b45f-062aa00df368",
                                                      "c4219558-db6c-476e-a9e5-81f4694f263c",
                                                      "next"});
}

TEST_CASE("4 subread split tagging", TEST_GROUP) {
    auto read = make_read();

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 3, messages);
    auto tag_node = pipeline_desc.add_node<dorado::SubreadTaggerNode>({sink}, 1, 1000);
    auto stereo_node = pipeline_desc.add_node<dorado::StereoDuplexEncoderNode>(
            {tag_node}, read->read_common.model_stride);
    auto pairing_node = pipeline_desc.add_node<dorado::PairingNode>(
            {stereo_node},
            dorado::DuplexPairingParameters{dorado::ReadOrder::BY_CHANNEL,
                                            dorado::DEFAULT_DUPLEX_CACHE_DEPTH},
            2, 1000);
    auto splitter = std::make_unique<const dorado::splitter::DuplexReadSplitter>(
            dorado::splitter::DuplexSplitSettings(false));
    pipeline_desc.add_node<dorado::ReadSplitNode>({pairing_node}, std::move(splitter), 1, 1000);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    pipeline->push_message(std::move(read));
    pipeline.reset();

    CHECK(messages.size() == 6);
    REQUIRE(std::all_of(messages.begin(), messages.end(),
                        [](const auto &message) { return is_read_message(message); }));

    std::vector<size_t> expected_subread_ids = {0, 1, 2, 3, 4, 5};
    std::vector<size_t> subread_ids;
    for (const auto &message : messages) {
        const auto &read_common = get_read_common_data(message);
        subread_ids.push_back(read_common.subread_id);
    }

    std::sort(std::begin(subread_ids), std::end(subread_ids));
    CHECK(subread_ids == expected_subread_ids);
    CHECK(std::all_of(messages.begin(), messages.end(),
                      [split_count = expected_subread_ids.size()](const auto &message) {
                          const auto &read_common = get_read_common_data(message);
                          return read_common.split_count == split_count;
                      }));
    CHECK(std::all_of(messages.begin(), messages.end(), [](const auto &message) {
        const auto &read_common = get_read_common_data(message);
        return read_common.read_tag == 42;
    }));
}

TEST_CASE("No split output read properties", TEST_GROUP) {
    const std::string init_read_id = "00a2dd45-f6a9-49ba-86ee-5d2a37b861cb";
    auto read = std::make_unique<dorado::SimplexRead>();
    read->range = 0;
    read->read_common.sample_rate = 4000;
    read->offset = -287;
    read->scaling = 0.14620706f;
    read->read_common.shift = 94.717316f;
    read->read_common.scale = 26.888939f;
    read->read_common.model_stride = 5;
    read->read_common.read_id = init_read_id;
    read->read_common.num_trimmed_samples = 10;
    read->read_common.attributes.read_number = 321;
    read->read_common.attributes.channel_number = 664;
    read->read_common.attributes.mux = 3;
    read->read_common.attributes.start_time = "2023-02-21T12:46:01.526+00:00";
    read->read_common.attributes.num_samples = 256790;
    read->start_sample = 29767426;
    read->end_sample = 30024216;
    read->run_acquisition_start_time_ms = 1676976119670;

    read->read_common.seq = "AAAAAAAAAAAAAAAAAAAAAA";
    read->read_common.qstring = std::string(read->read_common.seq.length(), '!');
    read->read_common.moves = std::vector<uint8_t>(read->read_common.seq.length(), 1);
    read->read_common.raw_data =
            at::zeros(read->read_common.seq.length() * 10).to(at::ScalarType::Half);
    read->read_common.read_tag = 42;
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 3, messages);
    auto splitter = std::make_unique<dorado::splitter::DuplexReadSplitter>(
            dorado::splitter::DuplexSplitSettings(false));
    pipeline_desc.add_node<dorado::ReadSplitNode>({sink}, std::move(splitter), 1, 1000);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    pipeline->push_message(std::move(read));
    pipeline.reset();

    auto reads = ConvertMessages<dorado::SimplexReadPtr>(std::move(messages));
    CHECK(reads.size() == 1);

    read = std::move(reads.front());
    CHECK(read->read_common.read_id == init_read_id);
    CHECK(read->read_common.subread_id == 0);
    CHECK(read->read_common.split_count == 1);
}

TEST_CASE("Test split where only one subread is generated", TEST_GROUP) {
    auto data_dir = std::filesystem::path(get_data_dir("split")) / "one_subread_split";

    auto read = std::make_unique<dorado::SimplexRead>();
    read->range = 0;
    read->read_common.sample_rate = 5000;
    read->offset = -260;
    read->scaling = 0.18707f;
    read->read_common.shift = 94.7565f;
    read->read_common.scale = 29.4467f;
    read->read_common.model_stride = 6;
    read->read_common.read_id = "6571a1d9-5dff-44f4-a526-558584ccea82";
    read->read_common.num_trimmed_samples = 4010;
    read->read_common.attributes.read_number = 10577;
    read->read_common.attributes.channel_number = 105;
    read->read_common.attributes.mux = 4;
    read->read_common.attributes.start_time = "2023-04-30T02:01:37.616+00:00";
    read->read_common.attributes.num_samples = 332541;
    read->start_sample = 178487546;
    read->end_sample = 178820087;
    read->run_acquisition_start_time_ms = 1682784400107;

    read->read_common.seq = ReadFileIntoString(data_dir / "seq");
    read->read_common.qstring = ReadFileIntoString(data_dir / "qstring");
    read->read_common.moves = ReadFileIntoVector(data_dir / "moves");
    torch::load(read->read_common.raw_data, (data_dir / "raw.tensor").string());
    read->read_common.raw_data = read->read_common.raw_data.to(at::ScalarType::Half);
    read->read_common.read_tag = 42;
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();

    dorado::PipelineDescriptor pipeline_desc;

    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 3, messages);
    auto splitter = std::make_unique<dorado::splitter::DuplexReadSplitter>(
            dorado::splitter::DuplexSplitSettings(false));
    pipeline_desc.add_node<dorado::ReadSplitNode>({sink}, std::move(splitter), 1, 1000);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    pipeline->push_message(std::move(read));
    pipeline.reset();

    CHECK(messages.size() == 1);

    const auto &read_common = get_read_common_data(messages[0]);
    CHECK(read_common.parent_read_id != read_common.read_id);
}
