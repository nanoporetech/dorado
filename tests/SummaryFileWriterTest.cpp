#include "hts_writer/SummaryFileWriter.h"

#include "hts_utils/hts_types.h"
#include "utils/string_utils.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <htslib/sam.h>

#include <sstream>
#include <string_view>

#define TEST_GROUP "[SummaryFileWriter]"

using namespace dorado;
using namespace std::string_view_literals;

CATCH_TEST_CASE(TEST_GROUP " constructor no throw", TEST_GROUP) {
    std::ostringstream stream;
    CATCH_CHECK_NOTHROW(hts_writer::SummaryFileWriter(stream, 0, std::nullopt));
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
    hts_writer::SummaryFileWriter writer(stream, flags, std::nullopt);
    auto columns = utils::split(stream.str(), '\t');
    CATCH_CHECK(columns.size() == expected_column_count);
}

CATCH_TEST_CASE(TEST_GROUP " rows have correct number of entries", TEST_GROUP) {
    using sfw = hts_writer::SummaryFileWriter;
    auto [flags] = GENERATE(table<sfw::FieldFlags>({
            {0},
            {sfw::BASECALLING_FIELDS | sfw::EXPERIMENT_FIELDS},
            {sfw::BARCODING_FIELDS | sfw::POLYA_FIELDS | sfw::DUPLEX_FIELDS},
            {sfw::BARCODING_FIELDS | sfw::ALIGNMENT_FIELDS | sfw::BASECALLING_FIELDS},
    }));
    std::ostringstream stream;
    hts_writer::SummaryFileWriter writer(stream, flags, std::nullopt);

    BamPtr bam_record(bam_init1());
    std::string_view record_name = "test_record"sv;
    std::string_view seq = "ACGT"sv;
    std::string_view qual = "0000"sv;
    bam_set1(bam_record.get(), record_name.size(), record_name.data(), 4, -1, -1, 0, 0, nullptr, -1,
             -1, 0, seq.size(), seq.data(), qual.data(), 0);

    HtsData hts_data{
            std::move(bam_record),
            HtsData::ReadAttributes{.sequencing_kit = "TEST_KIT", .experiment_id = "TEST_EXP"},
            nullptr};
    auto item = std::ref(hts_data);
    writer.process(item);

    auto rows = utils::split(stream.str(), '\n');
    CATCH_REQUIRE(rows.size() == 3);  // header, record, newline
    rows.pop_back();                  // discard empty row at the end
    CATCH_CAPTURE(flags, rows.front());
    size_t expected_column_count = std::numeric_limits<size_t>::max();
    for (const auto& row : rows) {
        auto columns = utils::split(row, '\t');
        if (expected_column_count == std::numeric_limits<size_t>::max()) {
            expected_column_count = static_cast<int>(columns.size());
        } else {
            CATCH_CHECK(columns.size() == expected_column_count);
        }
        for (const auto& column : columns) {
            CATCH_CHECK_FALSE(column.empty());  // no empty entries
        }
    }
}
