#include "read_pipeline/ReadForwarderNode.h"

#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "[ReadForwarderNodeTest]"

CATCH_TEST_CASE("OnlyReadsExtracted", TEST_GROUP) {
    std::vector<dorado::Message> messages;
    auto add_to_vec_callback = [&messages](dorado::Message&& message) {
        messages.push_back(std::move(message));
    };

    dorado::PipelineDescriptor pipeline_desc;
    pipeline_desc.add_node<dorado::ReadForwarderNode>({}, 10, 1, add_to_vec_callback);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    // Test that only simplex and duplex reads are passed out of ReadForwarderNode.
    pipeline->push_message(std::make_unique<dorado::SimplexRead>());
    pipeline->push_message(dorado::BamMessage());
    pipeline->push_message(dorado::ReadPair());
    pipeline->push_message(dorado::CacheFlushMessage());
    pipeline->push_message(std::make_unique<dorado::DuplexRead>());
    pipeline.reset();

    CATCH_CHECK(messages.size() == 2);
}