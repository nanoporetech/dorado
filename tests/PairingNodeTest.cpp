#include "read_pipeline/PairingNode.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/DuplexSplitNode.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>
#include <torch/torch.h>

#define TEST_GROUP "[PairingNodeTest]"

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
    read->start_time_ms = read->run_acquisition_start_time_ms +
                          uint64_t(std::round(read->start_sample * 1000. / read->sample_rate));
    read->seq = ReadFileIntoString(DataPath("seq"));
    read->qstring = ReadFileIntoString(DataPath("qstring"));
    read->moves = ReadFileIntoVector(DataPath("moves"));
    torch::load(read->raw_data, DataPath("raw.tensor").string());
    read->raw_data = read->raw_data.to(torch::kFloat16);
    return read;
}
}  // namespace

TEST_CASE("Split read pairing", TEST_GROUP) {
    auto read = make_read();

    MessageSinkToVector<std::shared_ptr<dorado::Read>> sink1(5);
    dorado::DuplexSplitSettings splitter_settings;
    dorado::DuplexSplitNode splitter_node(sink1, splitter_settings, 1);
    splitter_node.push_message(std::move(read));
    splitter_node.terminate();
    auto reads = sink1.get_messages();
    // this read splits into 4
    CHECK(reads.size() == 4);

    MessageSinkToVector<dorado::Message> sink2(5);
    dorado::PairingNode pairing_node(sink2);
    for (auto& read : reads) {
        pairing_node.push_message(std::move(read));
    }
    pairing_node.terminate();
    auto messages = sink2.get_messages();
    // the 4 split reads generate one additional readpair
    CHECK(messages.size() == 5);
    auto num_reads =
            std::count_if(messages.begin(), messages.end(), [](const dorado::Message& message) {
                return std::holds_alternative<std::shared_ptr<dorado::Read>>(message);
            });
    CHECK(num_reads == 4);
    auto num_pairs =
            std::count_if(messages.begin(), messages.end(), [](const dorado::Message& message) {
                return std::holds_alternative<std::shared_ptr<dorado::ReadPair>>(message);
            });
    CHECK(num_pairs == 1);
}
