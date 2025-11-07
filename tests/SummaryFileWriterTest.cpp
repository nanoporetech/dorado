#include "hts_writer/SummaryFileWriter.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <sstream>

#define TEST_GROUP "[SummaryFileWriter]"

using namespace dorado;

CATCH_TEST_CASE(TEST_GROUP " constructor no throw", TEST_GROUP) {
    std::ostringstream stream;
    CATCH_CHECK_NOTHROW(hts_writer::SummaryFileWriter(stream, 0));
}
