#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/ResumeLoaderNode.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[read_pipeline][ResumeLoaderNode]"

namespace fs = std::filesystem;

TEST_CASE(TEST_GROUP) {
    MessageSinkToVector<std::shared_ptr<dorado::Read>> sink(100);
    fs::path aligner_test_dir = fs::path(get_data_dir("aligner_test"));
    auto sam = aligner_test_dir / "basecall.sam";

    dorado::ResumeLoaderNode loader(sink, sam);
    loader.copy_completed_reads();
    CHECK(sink.get_messages().size() == 1);
    auto read_ids = loader.get_processed_read_ids();
    CHECK(read_ids.find("002bd127-db82-436f-b828-28567c3d505d") != read_ids.end());
}
