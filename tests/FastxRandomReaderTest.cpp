#include "hts_io/FastxRandomReader.h"

#include "TestUtils.h"
#include "read_pipeline/HtsWriter.h"
#include "utils/bam_utils.h"
#include "utils/hts_file.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/sam.h>

namespace {

using namespace dorado;

BamPtr generate_bam_entry(const std::string& read_id,
                          const std::string& seq,
                          const std::vector<uint8_t>& qscore) {
    BamPtr rec = BamPtr(bam_init1());

    const char* quals =
            std::empty(qscore) ? nullptr : reinterpret_cast<const char*>(std::data(qscore));

    bam_set1(rec.get(), read_id.length(), read_id.c_str(), 4, -1, -1, 0, 0, nullptr, -1, -1, 0,
             seq.length(), seq.c_str(), quals, 0);
    return rec;
}

}  // namespace

CATCH_TEST_CASE("Check if a read can be loaded correctly from FASTA input.", "FastxRandomReader") {
    auto temp_dir = tests::make_temp_dir("fastx_random_reader_test");
    auto temp_input_file = temp_dir.m_path / "input.fasta";

    const std::string seq = "ACTGATCG";
    const std::string read_id = "read1";

    // Write temporary file.
    {
        utils::HtsFile hts_file(temp_input_file.string(), utils::HtsFile::OutputMode::FASTA, 2,
                                false);
        HtsWriter writer(hts_file, "");
        auto rec = generate_bam_entry(read_id, seq, {});
        writer.write(rec.get());
        hts_file.finalise([](size_t) { /* noop */ });
    }

    hts_io::FastxRandomReader reader(temp_input_file.string());
    CATCH_CHECK(reader.num_entries() == 1);
    CATCH_CHECK(reader.fetch_seq(read_id) == seq);
}

CATCH_TEST_CASE("Check if a read can be loaded correctly from FASTQ input.", "FastxRandomReader") {
    auto temp_dir = tests::make_temp_dir("fastx_random_reader_test");
    auto temp_input_file = temp_dir.m_path / "input.fq";

    const std::string seq = "ACTGATCG";
    const std::vector<uint8_t> qscore = {20, 20, 30, 30, 20, 20, 40, 40};
    const std::string read_id = "read1";

    // Write temporary file.
    {
        utils::HtsFile hts_file(temp_input_file.string(), utils::HtsFile::OutputMode::FASTQ, 2,
                                false);
        HtsWriter writer(hts_file, "");
        auto rec = generate_bam_entry(read_id, seq, qscore);
        writer.write(rec.get());
        hts_file.finalise([](size_t) { /* noop */ });
    }

    hts_io::FastxRandomReader reader(temp_input_file.string());
    CATCH_CHECK(reader.num_entries() == 1);
    CATCH_CHECK(reader.fetch_seq(read_id) == seq);
    CATCH_CHECK(reader.fetch_qual(read_id) == qscore);
}
