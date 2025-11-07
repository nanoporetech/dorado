#include "hts_writer/SummaryFileWriter.h"

#include "utils/string_utils.h"

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

CATCH_TEST_CASE(TEST_GROUP " column headers by flags", TEST_GROUP) {
    using sfw = hts_writer::SummaryFileWriter;
    auto [flags, expected_column_count] = GENERATE(table<sfw::FieldFlags, size_t>({
            {0, 10},
            {sfw::BASECALLING_FIELDS | sfw::EXPERIMENT_FIELDS, 20},
            {sfw::BARCODING_FIELDS | sfw::POLYA_FIELDS | sfw::DUPLEX_FIELDS, 29},
            {sfw::BARCODING_FIELDS | sfw::ALIGNMENT_FIELDS | sfw::BASECALLING_FIELDS, 47},
    }));
    std::ostringstream stream;
    hts_writer::SummaryFileWriter writer(stream, flags);
    auto columns = utils::split(stream.str(), '\t');
    CATCH_CHECK(columns.size() == expected_column_count);
}
