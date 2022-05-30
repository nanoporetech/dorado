#include "TestUtils.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "Fast5DataLoaderTest: "

class MockSink : public ReadSink {
public:
    MockSink() : ReadSink(1000) {}
    size_t get_read_count() { return m_reads.size(); }
};

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 files") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    std::string data_path(get_data_dir());
    DataLoader loader(mock_sink, "cpu");
    loader.load_reads(data_path);

    REQUIRE(mock_sink.get_read_count() == 1);
}