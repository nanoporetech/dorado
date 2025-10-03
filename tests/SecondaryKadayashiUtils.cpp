#include "TestUtils.h"
#include "secondary/features/kadayashi_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace dorado::secondary::kadayashi_utils::tests {

#define TEST_GROUP "[SecondaryKadayashiUtils]"

namespace {
/**
 * \brief Helper function to load the haplotags from a .tsv file to compare as truth.
 */
std::unordered_map<std::string, int32_t> load_from_tsv(const std::filesystem::path& in_fn) {
    std::unordered_map<std::string, int32_t> ret;
    std::ifstream ifs(in_fn);
    std::string line;
    while (std::getline(ifs, line)) {
        if (std::empty(line)) {
            break;
        }
        if (line[0] != 'R') {
            continue;
        }

        // Parse the haptag.
        std::istringstream iss(line);
        std::string key;
        std::string ck;
        std::string qname;
        int32_t haptag = 0;
        iss >> key >> ck >> qname >> haptag;
        ret[qname] = haptag;
    }
    return ret;
}
}  // namespace

CATCH_TEST_CASE("Nonexistent input path, should throw", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-01-kadayashi-parser";
    const std::filesystem::path in_bin_fn = "nonexistent.bin";

    // Results.
    CATCH_CHECK_THROWS(query_bin_file_get_qname2tag(in_bin_fn, "chr20", 0, 10000));
}

CATCH_TEST_CASE("Start coordinate < 0, should return empty", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-01-kadayashi-parser";
    const std::filesystem::path in_bin_fn = test_data_dir / "in.phased.bin";

    // Results.
    const std::unordered_map<std::string, int32_t> result =
            query_bin_file_get_qname2tag(in_bin_fn, "chr20", -1, 10000);

    CATCH_CHECK(std::empty(result));
}

CATCH_TEST_CASE("End coordinate <= 0, should return empty", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-01-kadayashi-parser";
    const std::filesystem::path in_bin_fn = test_data_dir / "in.phased.bin";

    // Check the case when the end coordinate is zero.
    {
        const std::unordered_map<std::string, int32_t> result =
                query_bin_file_get_qname2tag(in_bin_fn, "chr20", 0, 0);
        CATCH_CHECK(std::empty(result));
    }

    // Check the case when the end coordinate is negative.
    {
        const std::unordered_map<std::string, int32_t> result =
                query_bin_file_get_qname2tag(in_bin_fn, "chr20", 0, -1);
        CATCH_CHECK(std::empty(result));
    }
}

CATCH_TEST_CASE("Missing reference, should return empty", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-01-kadayashi-parser";
    const std::filesystem::path in_bin_fn = test_data_dir / "in.phased.bin";

    // Results.
    const std::unordered_map<std::string, int32_t> result =
            query_bin_file_get_qname2tag(in_bin_fn, "nonexistent", 0, 10000);

    CATCH_CHECK(std::empty(result));
}

CATCH_TEST_CASE("Input file is truncated and reading fails, should throw", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-01-kadayashi-parser";
    const std::filesystem::path in_bin_fn = test_data_dir / "in.phased.bin";

    // Temporary location for a generated truncated file.
    const TempDir tmp_dir = make_temp_dir("kadayashi_utils_out");
    const std::filesystem::path in_truncated_bin_fn = tmp_dir.m_path / "truncated.bin";

    // Create a truncated input file.
    {
        // Open the input file and find out how big it is.
        std::ifstream ifs(in_bin_fn, std::ios::binary | std::ios::ate);
        const std::ifstream::pos_type file_size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        // Take 70% of the input file.
        const int64_t num_bytes_to_copy = static_cast<int64_t>(file_size) * 70 / 100;
        std::vector<char> buffer(num_bytes_to_copy);
        ifs.read(std::data(buffer), num_bytes_to_copy);

        // Write the truncated file.
        std::ofstream ofs(in_truncated_bin_fn, std::ios::binary | std::ios::trunc);
        ofs.write(std::data(buffer), ifs.gcount());
    }

    CATCH_CHECK_THROWS(query_bin_file_get_qname2tag(in_truncated_bin_fn, "chr20", 0, 10000));
}

CATCH_TEST_CASE("Load phasing_bin", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-01-kadayashi-parser";
    const std::filesystem::path in_bin_fn = test_data_dir / "in.phased.bin";
    const std::filesystem::path in_tsv_fn = test_data_dir / "in.phased.tsv";

    // Expected results.
    const std::unordered_map<std::string, int32_t> expected = load_from_tsv(in_tsv_fn);
    std::vector<std::pair<std::string, int32_t>> expected_vec(std::begin(expected),
                                                              std::end(expected));
    std::sort(std::begin(expected_vec), std::end(expected_vec));

    // Actual results.
    const std::unordered_map<std::string, int32_t> result =
            query_bin_file_get_qname2tag(in_bin_fn, "chr20", 0, 10000);
    std::vector<std::pair<std::string, int32_t>> result_vec(std::begin(result), std::end(result));
    std::sort(std::begin(result_vec), std::end(result_vec));

    CATCH_CHECK(expected_vec == result_vec);
}

}  // namespace dorado::secondary::kadayashi_utils::tests