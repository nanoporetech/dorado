#include "read_pipeline/ReadFilterNode.h"

#include "MessageSinkUtils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[read_pipeline][ReadFilterNode]"

namespace {
auto make_filtered_pipeline(std::vector<dorado::Message>& messages,
                            size_t min_qscore,
                            size_t min_read_length,
                            std::unordered_set<std::string> reads_to_filter) {
    dorado::PipelineDescriptor pipeline_desc;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    pipeline_desc.add_node<dorado::ReadFilterNode>({sink}, min_qscore, min_read_length,
                                                   std::move(reads_to_filter), 2 /*threads*/);
    return dorado::Pipeline::create(std::move(pipeline_desc));
}
}  // namespace

TEST_CASE("ReadFilterNode: Filter read based on qscore", TEST_GROUP) {
    std::vector<dorado::Message> messages;
    {
        auto pipeline = make_filtered_pipeline(messages, 12, 0, {});

        std::shared_ptr<dorado::Read> read_1(new dorado::Read());
        read_1->raw_data = torch::empty(100);
        read_1->sample_rate = 4000;
        read_1->shift = 128.3842f;
        read_1->scale = 8.258f;
        read_1->read_id = "read_1";
        read_1->seq = "ACGTACGT";
        read_1->qstring = "********";  // average q score 9
        read_1->num_trimmed_samples = 132;
        read_1->attributes.mux = 2;
        read_1->attributes.read_number = 18501;
        read_1->attributes.channel_number = 5;
        read_1->attributes.start_time = "2017-04-29T09:10:04Z";
        read_1->attributes.fast5_filename = "batch_0.fast5";

        std::shared_ptr<dorado::Read> read_2(new dorado::Read());
        read_2->raw_data = torch::empty(100);
        read_2->sample_rate = 4000;
        read_2->shift = 128.3842f;
        read_2->scale = 8.258f;
        read_2->read_id = "read_2";
        read_2->seq = "ACGTACGT";
        read_2->qstring = "////////";  // average q score 14
        read_2->num_trimmed_samples = 132;
        read_2->attributes.mux = 2;
        read_2->attributes.read_number = 18501;
        read_2->attributes.channel_number = 5;
        read_2->attributes.start_time = "2017-04-29T09:10:04Z";
        read_2->attributes.fast5_filename = "batch_0.fast5";

        pipeline->push_message(std::move(read_1));
        pipeline->push_message(std::move(read_2));
    }

    auto reads = ConvertMessages<std::shared_ptr<dorado::Read>>(std::move(messages));
    REQUIRE(reads.size() == 1);
    CHECK(reads[0]->read_id == "read_2");
}

TEST_CASE("ReadFilterNode: Filter read based on read name", TEST_GROUP) {
    std::vector<dorado::Message> messages;
    {
        auto pipeline = make_filtered_pipeline(messages, 0, 0, {"read_2"});

        std::shared_ptr<dorado::Read> read_1(new dorado::Read());
        read_1->raw_data = torch::empty(100);
        read_1->sample_rate = 4000;
        read_1->shift = 128.3842f;
        read_1->scale = 8.258f;
        read_1->read_id = "read_1";
        read_1->seq = "ACGTACGT";
        read_1->qstring = "********";  // average q score 9
        read_1->num_trimmed_samples = 132;
        read_1->attributes.mux = 2;
        read_1->attributes.read_number = 18501;
        read_1->attributes.channel_number = 5;
        read_1->attributes.start_time = "2017-04-29T09:10:04Z";
        read_1->attributes.fast5_filename = "batch_0.fast5";

        std::shared_ptr<dorado::Read> read_2(new dorado::Read());
        read_2->raw_data = torch::empty(100);
        read_2->sample_rate = 4000;
        read_2->shift = 128.3842f;
        read_2->scale = 8.258f;
        read_2->read_id = "read_2";
        read_2->seq = "ACGTACGT";
        read_2->qstring = "////////";  // average q score 14
        read_2->num_trimmed_samples = 132;
        read_2->attributes.mux = 2;
        read_2->attributes.read_number = 18501;
        read_2->attributes.channel_number = 5;
        read_2->attributes.start_time = "2017-04-29T09:10:04Z";
        read_2->attributes.fast5_filename = "batch_0.fast5";

        pipeline->push_message(std::move(read_1));
        pipeline->push_message(std::move(read_2));
    }

    auto reads = ConvertMessages<std::shared_ptr<dorado::Read>>(messages);
    REQUIRE(reads.size() == 1);
    CHECK(reads[0]->read_id == "read_1");
}

TEST_CASE("ReadFilterNode: Filter read based on read length", TEST_GROUP) {
    std::vector<dorado::Message> messages;
    {
        auto pipeline = make_filtered_pipeline(messages, 0, 5, {});

        std::shared_ptr<dorado::Read> read_1(new dorado::Read());
        read_1->raw_data = torch::empty(100);
        read_1->sample_rate = 4000;
        read_1->shift = 128.3842f;
        read_1->scale = 8.258f;
        read_1->read_id = "read_1";
        read_1->seq = "ACGTACGT";
        read_1->qstring = "********";  // average q score 9
        read_1->num_trimmed_samples = 132;
        read_1->attributes.mux = 2;
        read_1->attributes.read_number = 18501;
        read_1->attributes.channel_number = 5;
        read_1->attributes.start_time = "2017-04-29T09:10:04Z";
        read_1->attributes.fast5_filename = "batch_0.fast5";

        std::shared_ptr<dorado::Read> read_2(new dorado::Read());
        read_2->raw_data = torch::empty(100);
        read_2->sample_rate = 4000;
        read_2->shift = 128.3842f;
        read_2->scale = 8.258f;
        read_2->read_id = "read_2";
        read_2->seq = "ACGT";
        read_2->qstring = "////";  // average q score 14
        read_2->num_trimmed_samples = 132;
        read_2->attributes.mux = 2;
        read_2->attributes.read_number = 18501;
        read_2->attributes.channel_number = 5;
        read_2->attributes.start_time = "2017-04-29T09:10:04Z";
        read_2->attributes.fast5_filename = "batch_0.fast5";

        pipeline->push_message(std::move(read_1));
        pipeline->push_message(std::move(read_2));
    }

    auto reads = ConvertMessages<std::shared_ptr<dorado::Read>>(messages);
    REQUIRE(reads.size() == 1);
    CHECK(reads[0]->read_id == "read_1");
}
