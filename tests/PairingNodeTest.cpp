#include "read_pipeline/PairingNode.h"

#include "MessageSinkUtils.h"
#include "utils/time_utils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[PairingNodeTest]"

namespace {
std::shared_ptr<dorado::Read> make_read(int delay_ms, size_t seq_len) {
    std::shared_ptr<dorado::Read> read = std::make_shared<dorado::Read>();
    read->sample_rate = 4000;
    read->num_trimmed_samples = 10;
    read->attributes.channel_number = 664;
    read->attributes.mux = 3;
    read->attributes.num_samples = 10000;
    read->start_sample = 29767426 + (delay_ms * read->sample_rate) / 1000;
    read->end_sample = read->start_sample + read->attributes.num_samples;
    read->run_acquisition_start_time_ms = 1676976119670;
    read->start_time_ms = read->run_acquisition_start_time_ms +
                          uint64_t(std::round(read->start_sample * 1000. / read->sample_rate));
    read->attributes.start_time =
            dorado::utils::get_string_timestamp_from_unix_time(read->start_time_ms);
    read->seq = std::string(seq_len, 'A');
    return read;
}
}  // namespace

TEST_CASE("Split read pairing", TEST_GROUP) {
    // reads are 2500ms long
    // the second read must start within 5000ms of the end of the first read
    // and min/max length ratio must be greater than 0.95
    // expected pairs: {2, 3} only

    // clang-format off
    std::vector<std::shared_ptr<dorado::Read>> reads = {
            make_read(0, 1000),
            make_read(10, 1000),     // too early to pair with {0}
            make_read(10000, 1000),  // too late to pair with {1}
            make_read(15000, 990),
            make_read(18000, 800)    // too short to pair with {2}
    };
    // clang-format on

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 5, messages);
    // one thread, one read - force reads through in order
    auto pairing_node = pipeline_desc.add_node<dorado::PairingNode>(
            {sink}, dorado::ReadOrder::BY_CHANNEL, 1, 1);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    for (auto& read : reads) {
        pipeline->push_message(std::move(read));
    }
    pipeline.reset();

    // the 4 split reads generate one additional readpair
    CHECK(messages.size() == 6);
    auto num_reads =
            std::count_if(messages.begin(), messages.end(), [](const dorado::Message& message) {
                return std::holds_alternative<std::shared_ptr<dorado::Read>>(message);
            });
    CHECK(num_reads == 5);
    auto num_pairs =
            std::count_if(messages.begin(), messages.end(), [](const dorado::Message& message) {
                return std::holds_alternative<std::shared_ptr<dorado::ReadPair>>(message);
            });
    CHECK(num_pairs == 1);
}
