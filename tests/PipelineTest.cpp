#include "MessageSinkUtils.h"
#include "read_pipeline/NullNode.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "[Pipeline]"

using dorado::MessageSink;
using dorado::NodeHandle;
using dorado::NullNode;
using dorado::Pipeline;
using dorado::PipelineDescriptor;

CATCH_TEST_CASE("Creation", TEST_GROUP) {
    {
        // Empty pipelines are not allowed.
        PipelineDescriptor pipeline_desc;
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline == nullptr);
    }

    {
        // A single node is allowed
        PipelineDescriptor pipeline_desc;
        pipeline_desc.add_node<NullNode>({});
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline != nullptr);
    }

    {
        // > 1 source node is not allowed.
        PipelineDescriptor pipeline_desc;
        pipeline_desc.add_node<NullNode>({});
        pipeline_desc.add_node<NullNode>({});
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline == nullptr);
    }

    {
        // 2 connected nodes, with 1 a source is allowed.
        PipelineDescriptor pipeline_desc;
        auto sink = pipeline_desc.add_node<NullNode>({});
        pipeline_desc.add_node<NullNode>({sink});
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline != nullptr);
    }

    {
        // Sinks can be specified after construction.
        PipelineDescriptor pipeline_desc;
        auto sink = pipeline_desc.add_node<NullNode>({});
        auto source = pipeline_desc.add_node<NullNode>({});
        pipeline_desc.add_node_sink(source, sink);
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline != nullptr);
    }

    {
        // Directed cycles are not allowed.
        PipelineDescriptor pipeline_desc;
        auto a = pipeline_desc.add_node<NullNode>({});
        auto b = pipeline_desc.add_node<NullNode>({a});
        pipeline_desc.add_node_sink(a, b);
        pipeline_desc.add_node<NullNode>({a});
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline == nullptr);
    }

    {
        // Branching graph is allowed.
        PipelineDescriptor pipeline_desc;
        auto sink_a = pipeline_desc.add_node<NullNode>({});
        auto sink_b = pipeline_desc.add_node<NullNode>({});
        pipeline_desc.add_node<NullNode>({sink_a, sink_b});
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline != nullptr);
    }

    {
        // Undirected cycles are allowed.
        PipelineDescriptor pipeline_desc;
        auto sink_c = pipeline_desc.add_node<NullNode>({});
        auto sink_a = pipeline_desc.add_node<NullNode>({sink_c});
        auto sink_b = pipeline_desc.add_node<NullNode>({sink_c});
        pipeline_desc.add_node<NullNode>({sink_a, sink_b});
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_CHECK(pipeline != nullptr);
    }
}

// Tests destruction order of a random linear pipeline.
CATCH_TEST_CASE("LinearDestructionOrder", TEST_GROUP) {
    // Node that records destruction order.
    class OrderTestNode : public MessageSink {
    public:
        OrderTestNode(int index, std::vector<NodeHandle>& destruction_order)
                : MessageSink(1, 0), m_destruction_order(destruction_order), m_index(index) {}
        ~OrderTestNode() { m_destruction_order.push_back(m_index); }
        std::string get_name() const override { return "OrderTestNode"; }
        void terminate(const dorado::FlushOptions&) override {}
        void restart() override {}

    private:
        std::vector<int>& m_destruction_order;
        int m_index;
    };

    PipelineDescriptor pipeline_desc;
    std::vector<NodeHandle> handles;
    std::vector<int> indices;
    std::vector<int> destruction_order;
    const int kNumNodes = 10;
    for (int i = 0; i < kNumNodes; ++i) {
        handles.push_back(pipeline_desc.add_node<OrderTestNode>({}, i, destruction_order));
        indices.push_back(i);
    }

    // Shuffle handles / indices in a consistent manner, so we have index ordering
    // consistent with handle ordering.
    const int kNumSwaps = 100;
    for (int i = 0; i < kNumSwaps; ++i) {
        int a = rand() % kNumNodes;
        int b = rand() % kNumNodes;
        std::swap(handles.at(a), handles.at(b));
        std::swap(indices.at(a), indices.at(b));
    }

    // Construct a linear pipeline in the order of our shuffled handles.
    for (int i = 0; i < kNumNodes - 1; ++i) {
        pipeline_desc.add_node_sink(handles.at(i), handles.at(i + 1));
    }

    // Create and destroy a pipeline in our specified order.
    auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);
    pipeline.reset();

    // Verify that nodes were destroyed in source-to-sink order.
    CATCH_CHECK(std::equal(destruction_order.cbegin(), destruction_order.cend(), indices.cbegin()));
}

// Test inputs flow in the expected way from the source node.
CATCH_TEST_CASE("PipelineFlow", TEST_GROUP) {
    // NullNode passes nothing on, so the sink should get no messages
    // if they are sent to that node first.
    {
        // Natural construction order: sink to source.
        PipelineDescriptor pipeline_desc;
        std::vector<dorado::Message> messages;
        auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
        pipeline_desc.add_node<NullNode>({sink});
        auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_REQUIRE(pipeline != nullptr);
        pipeline->push_message(std::make_unique<dorado::SimplexRead>());
        pipeline.reset();
        CATCH_CHECK(messages.size() == 0);
    }

    {
        // Perverse construction order: source to sink.
        PipelineDescriptor pipeline_desc;
        std::vector<dorado::Message> messages;
        auto null_node = pipeline_desc.add_node<NullNode>({});
        auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
        pipeline_desc.add_node_sink(null_node, sink);
        auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);
        CATCH_REQUIRE(pipeline != nullptr);
        pipeline->push_message(std::make_unique<dorado::SimplexRead>());
        pipeline.reset();
        CATCH_CHECK(messages.size() == 0);
    }
}

// Test reads make it through after the pipeline has been restarted.
CATCH_TEST_CASE("TerminateRestart", TEST_GROUP) {
    PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);
    CATCH_REQUIRE(pipeline != nullptr);
    pipeline->push_message(std::make_unique<dorado::SimplexRead>());
    pipeline->terminate(dorado::DefaultFlushOptions());
    // Messages should get through as soon as we have terminated.
    CATCH_CHECK(messages.size() == 1);
    // If we restart we should be able to get another message through.
    pipeline->restart();
    pipeline->push_message(std::make_unique<dorado::SimplexRead>());
    pipeline.reset();
    CATCH_CHECK(messages.size() == 2);
}