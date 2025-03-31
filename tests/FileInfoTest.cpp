#include "TestUtils.h"
#include "file_info/file_info.h"
#include "utils/fs_utils.h"
#include "utils/stream_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#define TEST_GROUP "[dorado::file_info]"

namespace dorado::file_info::test {

namespace {
const auto& dir_entries = utils::fetch_directory_entries;
}

CATCH_TEST_CASE(TEST_GROUP "Test calculating number of reads from fast5, read ids list.",
                TEST_GROUP) {
    auto data_path = get_fast5_data_dir();
    const auto folder_entries = dir_entries(data_path, false);
    CATCH_SECTION("fast5 file only, no read ids list") {
        CATCH_CHECK(get_num_reads(folder_entries, std::nullopt, {}) == 1);
    }

    CATCH_SECTION("fast5 file and read ids with 0 reads") {
        auto read_list = std::unordered_set<std::string>();
        CATCH_CHECK(get_num_reads(folder_entries, read_list, {}) == 0);
    }
    CATCH_SECTION("fast5 file and read ids with 2 reads") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("1");
        read_list.insert("2");
        CATCH_CHECK(get_num_reads(folder_entries, read_list, {}) == 1);
    }
}

CATCH_TEST_CASE(TEST_GROUP "Find sample rate from fast5", TEST_GROUP) {
    auto data_path = get_fast5_data_dir();
    CATCH_CHECK(get_sample_rate(dir_entries(data_path, false)) == 6024);
}

CATCH_TEST_CASE(TEST_GROUP "Test calculating number of reads from pod5, read ids list.",
                TEST_GROUP) {
    auto data_path = get_pod5_data_dir();
    const auto folder_entries = dir_entries(data_path, false);
    CATCH_SECTION("pod5 file only, no read ids list") {
        CATCH_CHECK(get_num_reads(folder_entries, std::nullopt, {}) == 1);
    }

    CATCH_SECTION("pod5 file and read ids with 0 reads") {
        auto read_list = std::unordered_set<std::string>();
        CATCH_CHECK(get_num_reads(folder_entries, read_list, {}) == 0);
    }
    CATCH_SECTION("pod5 file and read ids with 2 reads") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("1");
        read_list.insert("2");
        CATCH_CHECK(get_num_reads(folder_entries, read_list, {}) == 1);
    }
}

CATCH_TEST_CASE(TEST_GROUP "Find sample rate from single pod5.", TEST_GROUP) {
    auto single_read_path = get_pod5_data_dir();
    CATCH_CHECK(get_sample_rate(dir_entries(single_read_path, false)) == 4000);
}

CATCH_TEST_CASE(TEST_GROUP "Find sample rate from pod5 dir.", TEST_GROUP) {
    auto data_path = get_pod5_data_dir();
    CATCH_CHECK(get_sample_rate(dir_entries(data_path, false)) == 4000);
}

CATCH_TEST_CASE(TEST_GROUP "Find sample rate from nested pod5.", TEST_GROUP) {
    auto data_path = get_nested_pod5_data_dir();
    CATCH_CHECK(get_sample_rate(dir_entries(data_path, true)) == 4000);
}

CATCH_TEST_CASE(TEST_GROUP "Test loading POD5 file with read ignore list", TEST_GROUP) {
    auto data_path = get_data_dir("multi_read_pod5");
    const auto folder_entries = dir_entries(data_path, false);
    CATCH_SECTION("read ignore list with 1 read") {
        auto read_ignore_list = std::unordered_set<std::string>();
        read_ignore_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5
        CATCH_CHECK(get_num_reads(folder_entries, std::nullopt, read_ignore_list) == 3);
    }

    CATCH_SECTION("same read in read_ids and ignore list") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5
        auto read_ignore_list = std::unordered_set<std::string>();
        read_ignore_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5
        CATCH_CHECK(get_num_reads(folder_entries, read_list, read_ignore_list) == 0);
    }
}

CATCH_TEST_CASE(TEST_GROUP "  get_unique_sequencing_chemistry", TEST_GROUP) {
    using CC = models::Chemistry;
    namespace fs = std::filesystem;

    CATCH_SECTION("get_chemistry from homogeneous datasets") {
        auto [condition, expected] = GENERATE(table<std::string, CC>({
                std::make_tuple("dna_r10.4.1_e8.2_260bps", CC::DNA_R10_4_1_E8_2_260BPS),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_4khz", CC::DNA_R10_4_1_E8_2_400BPS_4KHZ),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_5khz", CC::DNA_R10_4_1_E8_2_400BPS_5KHZ),
                std::make_tuple("dna_r9.4.1_e8", CC::DNA_R9_4_1_E8),
                std::make_tuple("rna002_70bps", CC::RNA002_70BPS),
                std::make_tuple("rna004_130bps", CC::RNA004_130BPS),
        }));

        CATCH_CAPTURE(condition);
        auto data = fs::path(get_data_dir("pod5")) / condition;
        CATCH_CHECK(fs::exists(data));
        auto result = get_unique_sequencing_chemistry(dir_entries(data.u8string(), false));
        CATCH_CHECK(result == expected);
    }

    CATCH_SECTION("get_chemistry throws with inhomogeneous") {
        auto data = fs::path(get_data_dir("pod5")) / "mixed";
        utils::SuppressStdout suppress_error_message{};
        CATCH_CHECK_THROWS(get_unique_sequencing_chemistry(dir_entries(data.u8string(), true)),
                           Catch::Matchers::Contains(
                                   "Could not uniquely resolve chemistry from inhomogeneous data"));
    }
}

}  // namespace dorado::file_info::test