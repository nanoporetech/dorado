#include "hts_utils/FastxSequentialReader.h"

#include "TestUtils.h"

#include <catch2/catch_test_macros.hpp>

#include <fstream>

namespace {
void write_lines(const std::filesystem::path& path, const std::vector<std::string>& lines) {
    std::ofstream ofs(path);
    bool first = true;
    for (const std::string_view line : lines) {
        if (!first) {
            ofs << '\n';
        }
        ofs << line;
        first = false;
    }
}

std::vector<std::string> collect_results(dorado::hts_io::FastxSequentialReader& reader) {
    // Get and collect results.
    std::vector<std::string> results;
    dorado::hts_io::FastxRecord record;
    while (reader.get_next(record)) {
        if (std::empty(record.seq)) {
            continue;
        }
        const std::string header_leader = (std::empty(record.qual) ? ">" : "@");
        std::string header = header_leader + std::string{record.name};
        if (!std::empty(record.comment)) {
            header += " " + std::string{record.comment};
        }
        results.emplace_back(header);
        results.emplace_back(record.seq);
        if (!std::empty(record.qual)) {
            results.emplace_back("+");
            results.emplace_back(record.qual);
        }
    }
    return results;
}

}  // namespace

CATCH_TEST_CASE("FASTA input empty", "FastxSequentialReader") {
    using namespace dorado;

    auto temp_dir = tests::make_temp_dir("fastx_sequential_reader_test");
    auto temp_input_file = temp_dir.m_path / "input.fasta";

    const std::vector<std::string> input_data{};

    write_lines(temp_input_file, input_data);

    hts_io::FastxSequentialReader reader(temp_input_file);

    const std::vector<std::string> results = collect_results(reader);

    CATCH_CHECK(results == input_data);
}

CATCH_TEST_CASE("FASTA input full", "FastxSequentialReader") {
    using namespace dorado;

    auto temp_dir = tests::make_temp_dir("fastx_sequential_reader_test");
    auto temp_input_file = temp_dir.m_path / "input.fasta";

    const std::vector<std::string> input_data{
            ">read1 comment",
            "ACTG",
            ">read2 RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
            "ACTGATCG",
            ">read3",
            "ACUG",
    };

    write_lines(temp_input_file, input_data);

    hts_io::FastxSequentialReader reader(temp_input_file);

    const std::vector<std::string> results = collect_results(reader);

    CATCH_CHECK(results == input_data);
}

CATCH_TEST_CASE("FASTQ input empty", "FastxSequentialReader") {
    using namespace dorado;

    auto temp_dir = tests::make_temp_dir("fastx_sequential_reader_test");
    auto temp_input_file = temp_dir.m_path / "input.fastq";

    const std::vector<std::string> input_data{};

    write_lines(temp_input_file, input_data);

    hts_io::FastxSequentialReader reader(temp_input_file);

    const std::vector<std::string> results = collect_results(reader);

    CATCH_CHECK(results == input_data);
}

CATCH_TEST_CASE("FASTQ input", "FastxSequentialReader") {
    using namespace dorado;

    auto temp_dir = tests::make_temp_dir("fastx_sequential_reader_test");
    auto temp_input_file = temp_dir.m_path / "input.fastq";

    const std::vector<std::string> input_data{
            "@read1 comment",
            "ACTG",
            "+",
            "!!!!",
            "@read2 RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
            "ACTGATCG",
            "+",
            "++++++++",
            "@read3",
            "ACUG",
            "+",
            "@@@@",
    };

    write_lines(temp_input_file, input_data);

    hts_io::FastxSequentialReader reader(temp_input_file);

    const std::vector<std::string> results = collect_results(reader);

    CATCH_CHECK(results == input_data);
}
