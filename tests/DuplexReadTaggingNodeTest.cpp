#include "read_pipeline/DuplexReadTaggingNode.h"

#include "MessageSinkUtils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[read_pipeline][DuplexReadTaggingNode]"

TEST_CASE("DuplexReadTaggingNode", TEST_GROUP) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    pipeline_desc.add_node<dorado::DuplexReadTaggingNode>({sink});
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));
    {
        auto read_12 = std::make_shared<dorado::Read>();
        read_12->read_id = "1;2";
        read_12->is_duplex = true;

        auto read_1 = std::make_shared<dorado::Read>();
        read_1->read_id = "1";
        read_1->is_duplex_parent = true;

        auto read_2 = std::make_shared<dorado::Read>();
        read_2->read_id = "2";
        read_2->is_duplex_parent = true;

        auto read_3 = std::make_shared<dorado::Read>();
        read_3->read_id = "3";
        read_3->is_duplex_parent = true;

        auto read_4 = std::make_shared<dorado::Read>();
        read_4->read_id = "4";
        read_4->is_duplex_parent = true;

        auto read_5 = std::make_shared<dorado::Read>();
        read_5->read_id = "5";
        read_5->is_duplex_parent = true;

        auto read_6 = std::make_shared<dorado::Read>();
        read_6->read_id = "6";
        read_6->is_duplex_parent = true;

        auto read_56 = std::make_shared<dorado::Read>();
        read_6->read_id = "5;6";
        read_6->is_duplex = true;

        pipeline->push_message(read_1);
        pipeline->push_message(read_2);
        pipeline->push_message(read_3);
        pipeline->push_message(read_4);
        pipeline->push_message(read_12);
        pipeline->push_message(read_5);
        pipeline->push_message(read_6);
        pipeline->push_message(read_56);
    }
    pipeline.reset();

    auto reads = ConvertMessages<std::shared_ptr<dorado::Read>>(messages);
    for (auto& read : reads) {
        if (read->read_id == "1;2") {
            CHECK(read->is_duplex == true);
        }
        if (read->read_id == "1") {
            CHECK(read->is_duplex_parent == true);
        }
        if (read->read_id == "2") {
            CHECK(read->is_duplex_parent == true);
        }
        if (read->read_id == "3") {
            CHECK(read->is_duplex_parent == false);
        }
        if (read->read_id == "4") {
            CHECK(read->is_duplex == false);
        }
        if (read->read_id == "5") {
            CHECK(read->is_duplex_parent == true);
        }
        if (read->read_id == "6") {
            CHECK(read->is_duplex_parent == true);
        }
        if (read->read_id == "5;6") {
            CHECK(read->is_duplex == true);
        }
    }
}
