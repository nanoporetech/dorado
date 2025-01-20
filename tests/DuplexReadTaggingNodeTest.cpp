#include "read_pipeline/DuplexReadTaggingNode.h"

#include "MessageSinkUtils.h"

#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "[read_pipeline][DuplexReadTaggingNode]"

CATCH_TEST_CASE("DuplexReadTaggingNode", TEST_GROUP) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    pipeline_desc.add_node<dorado::DuplexReadTaggingNode>({sink});
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);
    {
        auto read_12 = std::make_unique<dorado::SimplexRead>();
        read_12->read_common.read_id = "1;2";
        read_12->read_common.is_duplex = true;

        auto read_1 = std::make_unique<dorado::SimplexRead>();
        read_1->read_common.read_id = "1";
        read_1->is_duplex_parent = true;

        auto read_2 = std::make_unique<dorado::SimplexRead>();
        read_2->read_common.read_id = "2";
        read_2->is_duplex_parent = true;

        auto read_3 = std::make_unique<dorado::SimplexRead>();
        read_3->read_common.read_id = "3";
        read_3->is_duplex_parent = true;

        auto read_4 = std::make_unique<dorado::SimplexRead>();
        read_4->read_common.read_id = "4";
        read_4->is_duplex_parent = true;

        auto read_5 = std::make_unique<dorado::SimplexRead>();
        read_5->read_common.read_id = "5";
        read_5->is_duplex_parent = true;

        auto read_6 = std::make_unique<dorado::SimplexRead>();
        read_6->read_common.read_id = "6";
        read_6->is_duplex_parent = true;

        auto read_56 = std::make_unique<dorado::SimplexRead>();
        read_6->read_common.read_id = "5;6";
        read_6->read_common.is_duplex = true;

        pipeline->push_message(std::move(read_1));
        pipeline->push_message(std::move(read_2));
        pipeline->push_message(std::move(read_3));
        pipeline->push_message(std::move(read_4));
        pipeline->push_message(std::move(read_12));
        pipeline->push_message(std::move(read_5));
        pipeline->push_message(std::move(read_6));
        pipeline->push_message(std::move(read_56));
    }
    pipeline.reset();

    auto reads = ConvertMessages<dorado::SimplexReadPtr>(std::move(messages));
    for (auto& read : reads) {
        if (read->read_common.read_id == "1;2") {
            CATCH_CHECK(read->read_common.is_duplex == true);
        }
        if (read->read_common.read_id == "1") {
            CATCH_CHECK(read->is_duplex_parent == true);
        }
        if (read->read_common.read_id == "2") {
            CATCH_CHECK(read->is_duplex_parent == true);
        }
        if (read->read_common.read_id == "3") {
            CATCH_CHECK(read->is_duplex_parent == false);
        }
        if (read->read_common.read_id == "4") {
            CATCH_CHECK(read->read_common.is_duplex == false);
        }
        if (read->read_common.read_id == "5") {
            CATCH_CHECK(read->is_duplex_parent == true);
        }
        if (read->read_common.read_id == "6") {
            CATCH_CHECK(read->is_duplex_parent == true);
        }
        if (read->read_common.read_id == "5;6") {
            CATCH_CHECK(read->read_common.is_duplex == true);
        }
    }
}
