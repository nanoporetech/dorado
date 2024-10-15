#include "TestUtils.h"
#include "file_info/file_info.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[FileInfoTest]"

TEST_CASE(TEST_GROUP "Test calculating number of reads from fast5, read ids list.", TEST_GROUP) {
    auto data_path = get_fast5_data_dir();

    SECTION("fast5 file only, no read ids list") {
        CHECK(dorado::file_info::get_num_reads(data_path, std::nullopt, {}, false) == 1);
    }

    SECTION("fast5 file and read ids with 0 reads") {
        auto read_list = std::unordered_set<std::string>();
        CHECK(dorado::file_info::get_num_reads(data_path, read_list, {}, false) == 0);
    }
    SECTION("fast5 file and read ids with 2 reads") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("1");
        read_list.insert("2");
        CHECK(dorado::file_info::get_num_reads(data_path, read_list, {}, false) == 1);
    }
}

TEST_CASE(TEST_GROUP "Find sample rate from fast5", TEST_GROUP) {
    auto data_path = get_fast5_data_dir();
    CHECK(dorado::file_info::get_sample_rate(data_path, false) == 6024);
}

TEST_CASE(TEST_GROUP "Test calculating number of reads from pod5, read ids list.", TEST_GROUP) {
    auto data_path = get_pod5_data_dir();

    SECTION("pod5 file only, no read ids list") {
        CHECK(dorado::file_info::get_num_reads(data_path, std::nullopt, {}, false) == 1);
    }

    SECTION("pod5 file and read ids with 0 reads") {
        auto read_list = std::unordered_set<std::string>();
        CHECK(dorado::file_info::get_num_reads(data_path, read_list, {}, false) == 0);
    }
    SECTION("pod5 file and read ids with 2 reads") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("1");
        read_list.insert("2");
        CHECK(dorado::file_info::get_num_reads(data_path, read_list, {}, false) == 1);
    }
}

TEST_CASE(TEST_GROUP "Find sample rate from single pod5.", TEST_GROUP) {
    auto single_read_path = get_pod5_data_dir();
    CHECK(dorado::file_info::get_sample_rate(single_read_path, false) == 4000);
}

TEST_CASE(TEST_GROUP "Find sample rate from pod5 dir.", TEST_GROUP) {
    auto data_path = get_pod5_data_dir();
    CHECK(dorado::file_info::get_sample_rate(data_path, false) == 4000);
}

TEST_CASE(TEST_GROUP "Find sample rate from nested pod5.", TEST_GROUP) {
    auto data_path = get_nested_pod5_data_dir();
    CHECK(dorado::file_info::get_sample_rate(data_path, true) == 4000);
}

TEST_CASE(TEST_GROUP "Test loading POD5 file with read ignore list", TEST_GROUP) {
    auto data_path = get_data_dir("multi_read_pod5");

    SECTION("read ignore list with 1 read") {
        auto read_ignore_list = std::unordered_set<std::string>();
        read_ignore_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5
        CHECK(dorado::file_info::get_num_reads(data_path, std::nullopt, read_ignore_list, false) ==
              3);
    }

    SECTION("same read in read_ids and ignore list") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5
        auto read_ignore_list = std::unordered_set<std::string>();
        read_ignore_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5
        CHECK(dorado::file_info::get_num_reads(data_path, read_list, read_ignore_list, false) == 0);
    }
}

TEST_CASE(TEST_GROUP "  get_unique_sequencing_chemisty", TEST_GROUP) {
    using CC = dorado::models::Chemistry;
    namespace fs = std::filesystem;

    SECTION("get_chemistry from homogeneous datasets") {
        auto [condition, expected] = GENERATE(table<std::string, CC>({
                std::make_tuple("dna_r10.4.1_e8.2_260bps", CC::DNA_R10_4_1_E8_2_260BPS),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_4khz", CC::DNA_R10_4_1_E8_2_400BPS_4KHZ),
                std::make_tuple("dna_r10.4.1_e8.2_400bps_5khz", CC::DNA_R10_4_1_E8_2_400BPS_5KHZ),
                std::make_tuple("dna_r9.4.1_e8", CC::DNA_R9_4_1_E8),
                std::make_tuple("rna002_70bps", CC::RNA002_70BPS),
                std::make_tuple("rna004_130bps", CC::RNA004_130BPS),
        }));

        CAPTURE(condition);
        auto data = fs::path(get_data_dir("pod5")) / condition;
        CHECK(fs::exists(data));
        auto result = dorado::file_info::get_unique_sequencing_chemisty(data.u8string(), false);
        CHECK(result == expected);
    }

    SECTION("get_chemistry throws with inhomogeneous") {
        auto data = fs::path(get_data_dir("pod5")) / "mixed";
        CHECK_THROWS(dorado::file_info::get_unique_sequencing_chemisty(data.u8string(), true),
                     Catch::Matchers::Contains(
                             "Could not uniquely resolve chemistry from inhomogeneous data"));
    }
}
