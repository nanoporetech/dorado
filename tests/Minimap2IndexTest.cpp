#include "alignment/Minimap2Index.h"

#include "TestUtils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[alignment::Minimap2Index]"

namespace fs = std::filesystem;

namespace dorado::alignment::test {

TEST_CASE(TEST_GROUP " newly constructed instance has zeroed indexing options", TEST_GROUP) {
    Minimap2Index cut{};
    REQUIRE(cut.index_options().k == 0);  // just confirm k is zero
}

TEST_CASE(TEST_GROUP " newly constructed instance has zeroed mapping options", TEST_GROUP) {
    Minimap2Index cut{};
    REQUIRE(cut.mapping_options().max_qlen == 0);  //just confirm max_qlen is zero
}

TEST_CASE(TEST_GROUP " initialise() with default options does not throw", TEST_GROUP) {
    Minimap2Index cut{};

    REQUIRE_NOTHROW(cut.initialise(dflt_options));
}

TEST_CASE(TEST_GROUP " initialise() with default options returns true", TEST_GROUP) {
    Minimap2Index cut{};

    REQUIRE(cut.initialise(dflt_options));
}

TEST_CASE(TEST_GROUP " initialise() with default options sets indexing options", TEST_GROUP) {
    Minimap2Index cut{};

    cut.initialise(dflt_options);

    REQUIRE(cut.index_options().k == dflt_options.kmer_size);
    REQUIRE(cut.index_options().w == dflt_options.window_size);
    REQUIRE(cut.index_options().batch_size == dflt_options.index_batch_size);
}

}  // namespace dorado::alignment::test
