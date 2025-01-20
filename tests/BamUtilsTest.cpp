#include "TestUtils.h"
#include "read_pipeline/HtsReader.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#define TEST_GROUP "[bam_utils]"

namespace fs = std::filesystem;
using namespace dorado;

CATCH_TEST_CASE("BamUtilsTest: fetch keys from PG header", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("aligner_test"));
    auto sam = aligner_test_dir / "basecall.sam";

    auto keys =
            utils::extract_pg_keys_from_hdr(sam.string(), {"PN", "CL", "VN"}, "ID", "basecaller");
    CATCH_CHECK(keys["PN"] == "dorado");
    CATCH_CHECK(keys["VN"] == "0.5.0+5fa4de73+dirty");
    CATCH_CHECK(
            keys["CL"] ==
            "dorado basecaller dna_r9.4.1_e8_hac@v3.3 ./tests/data/pod5 -x cpu --modified-bases "
            "5mCG --emit-sam");
}

CATCH_TEST_CASE("BamUtilsTest: Add read group headers scenarios", TEST_GROUP) {
    auto has_read_group_header = [](sam_hdr_t *ptr, const char *id) {
        return sam_hdr_line_index(ptr, "RG", id) >= 0;
    };
    KString barcode_kstring(1000000);
    auto get_barcode_tag = [&barcode_kstring](sam_hdr_t *ptr,
                                              const char *id) -> std::optional<std::string> {
        if (sam_hdr_find_tag_id(ptr, "RG", "ID", id, "BC", &barcode_kstring.get()) != 0) {
            return std::nullopt;
        }
        std::string tag(ks_str(&barcode_kstring.get()), ks_len(&barcode_kstring.get()));
        return tag;
    };

    CATCH_SECTION("No read groups generate no headers") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        CATCH_CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == 0);
        dorado::utils::add_rg_headers(sam_header.get(), {});
        CATCH_CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == 0);
    }

    const std::unordered_map<std::string, dorado::ReadGroup> read_groups{
            {"id_0",
             {"run_0", "basecalling_model_0", "modbase_model_0", "flowcell_0", "device_0",
              "exp_start_0", "sample_0", "", ""}},
            {"id_1",
             {"run_1", "basecalling_model_1", "modbase_model_1", "flowcell_1", "device_1",
              "exp_start_1", "sample_1", "", ""}},
    };

    CATCH_SECTION("Read groups") {
        dorado::SamHdrPtr sam_header(sam_hdr_init());
        dorado::utils::add_rg_headers(sam_header.get(), read_groups);

        // Check the IDs of the groups are all there.
        CATCH_CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == int(read_groups.size()));
        for (auto &&[id, read_group] : read_groups) {
            CATCH_CHECK(has_read_group_header(sam_header.get(), id.c_str()));
            // None of the read groups should have a barcode.
            CATCH_CHECK(get_barcode_tag(sam_header.get(), id.c_str()) == std::nullopt);
        }
    }

    CATCH_SECTION("Read groups with barcode kit") {
        const std::string CUSTOM_BARCODE_NAME{"CUSTOM-BC01"};
        const std::string CUSTOM_BARCODE_SEQUENCE{"AAA"};
        const std::string KIT_NAME{"CUSTOM-SQK-RAB204"};
        auto kit_info = barcode_kits::KitInfo{KIT_NAME, false,  false,
                                              false,    "ACGT", "ACGT",
                                              "ACGT",   "ACGT", {CUSTOM_BARCODE_NAME},
                                              {},       {}};
        dorado::SamHdrPtr sam_header(sam_hdr_init());

        std::unordered_map<std::string, std::string> custom_barcodes{
                {CUSTOM_BARCODE_NAME, CUSTOM_BARCODE_SEQUENCE}};

        barcode_kits::add_custom_barcode_kit(KIT_NAME, kit_info);
        auto kit_cleanup = dorado::utils::PostCondition(
                [] { dorado::barcode_kits::clear_custom_barcode_kits(); });
        barcode_kits::add_custom_barcodes(custom_barcodes);
        auto barcode_cleanup =
                dorado::utils::PostCondition([] { dorado::barcode_kits::clear_custom_barcodes(); });

        dorado::utils::add_rg_headers_with_barcode_kit(sam_header.get(), read_groups, KIT_NAME,
                                                       nullptr);

        // Check the IDs of the groups are all there.
        const size_t total_groups = read_groups.size() * (kit_info.barcodes.size() + 1);
        CATCH_CHECK(sam_hdr_count_lines(sam_header.get(), "RG") == int(total_groups));

        // Check that the IDs match the expected format.
        const auto &barcode_seqs = dorado::barcode_kits::get_barcodes();
        for (auto &&[id, read_group] : read_groups) {
            CATCH_CHECK(has_read_group_header(sam_header.get(), id.c_str()));
            CATCH_CHECK(get_barcode_tag(sam_header.get(), id.c_str()) == std::nullopt);

            // The headers with barcodes should contain those barcodes.
            for (const auto &barcode_name : kit_info.barcodes) {
                const auto full_id = id + "_" +
                                     dorado::barcode_kits::generate_standard_barcode_name(
                                             KIT_NAME, barcode_name);
                const auto &barcode_seq = barcode_seqs.at(barcode_name);
                CATCH_CHECK(has_read_group_header(sam_header.get(), full_id.c_str()));
                if (barcode_name != CUSTOM_BARCODE_NAME) {
                    CATCH_CHECK(get_barcode_tag(sam_header.get(), full_id.c_str()) == barcode_seq);
                }
            }

            // The custom barcode sequence should be present in the barcode tag
            const auto custom_full_id = id + "_" +
                                        dorado::barcode_kits::generate_standard_barcode_name(
                                                KIT_NAME, CUSTOM_BARCODE_NAME);
            auto actual_barcode_tag_sequence =
                    get_barcode_tag(sam_header.get(), custom_full_id.c_str());
            CATCH_CHECK(actual_barcode_tag_sequence == CUSTOM_BARCODE_SEQUENCE);
        }
    }
}

CATCH_TEST_CASE("BamUtilsTest: Test bam extraction helpers", TEST_GROUP) {
    fs::path bam_utils_test_dir = fs::path(get_data_dir("bam_utils"));
    auto sam = bam_utils_test_dir / "test.sam";

    HtsReader reader(sam.string(), std::nullopt);
    CATCH_REQUIRE(reader.read());  // Parse first and only record.
    auto record = reader.record.get();

    CATCH_SECTION("Test sequence extraction") {
        std::string seq = utils::extract_sequence(record);
        CATCH_CHECK(seq ==
                    "AATAAACCGAAGACAATTTAGAAGCCAGCGAGGTATGTGCGTCTACTTCGTTCGGTTATGCGAAGCCGATATAACCTG"
                    "CAGGAC"
                    "AACACAACATTTCCACTGTTTTCGTTCATTCGTAAACGCTTTCGCGTTCATCACACTCAACCATAGGCTTTAGCCAGA"
                    "ACGTTA"
                    "TGAACCCCAGCGACTTCCAGAACGGCGCGCGTGCCACCACCGGCGATGATACCGGTTCCTTCGGAAGCCGGCTGCATG"
                    "AATACG"
                    "CGAGAACCCGTGTGAACACCTTTAACAGGGTGTTGCAGAGTGCCGTTGCTGCGGCACGATAGTTAAGTCGTATTGCTG"
                    "AAGCGA"
                    "CACTGTCCATCGCTTTCTGGATGGCT");
    }

    CATCH_SECTION("Test quality extraction") {
        const std::string qual =
                "%$%&%$####%'%%$&'(1/...022.+%%%%%%$$%%&%$%%%&&+)()./"
                "0%$$'&'&'%$###$&&&'*(()()%%%%(%%'))(('''3222276<BAAABE:+''&)**%(/"
                "''(:322**(*,,++&+++/1)(&&(006=B??@AKLK=<==HHHHHFFCBB@??>==943323/-.'56::71.//"
                "0933))%&%&))*1739:666455116/"
                "0,(%%&(*-55EBEB>@;??>>@BBDC?><<98-,,BGHEGFFGIIJFFDBB;6AJ>===KB:::<70/"
                "..--,++,))+*)&&'*-,+*)))(%%&'&''%%%$&%$###$%%$$%'%%$$+1.--.7969....*)))";
        auto qual_vector = utils::extract_quality(record);
        CATCH_CHECK(qual_vector.size() == qual.length());
        for (size_t i = 0; i < qual.length(); i++) {
            CATCH_CHECK(qual[i] == qual_vector[i] + 33);
        }
    }

    CATCH_SECTION("Test move table extraction") {
        auto [stride, move_table] = utils::extract_move_table(record);
        int seqlen = record->core.l_qseq;
        CATCH_REQUIRE(!move_table.empty());
        CATCH_CHECK(stride == 6);
        CATCH_CHECK(seqlen == std::accumulate(move_table.begin(), move_table.end(), 0));
    }

    CATCH_SECTION("Test mod base info extraction") {
        auto [modbase_str, modbase_probs] = utils::extract_modbase_info(record);
        const std::vector<int8_t> expected_modbase_probs = {5, 1};
        CATCH_CHECK(modbase_str == "C+h?,1;C+m?,1;");
        CATCH_CHECK(modbase_probs.size() == expected_modbase_probs.size());
        for (size_t i = 0; i < expected_modbase_probs.size(); i++) {
            CATCH_CHECK(modbase_probs[i] == expected_modbase_probs[i]);
        }
    }
}

CATCH_TEST_CASE("BamUtilsTest: cigar2str utility", TEST_GROUP) {
    const std::string cigar = "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S";
    size_t m = 0;
    uint32_t *a_cigar = NULL;
    char *end = NULL;
    int n_cigar = int(sam_parse_cigar(cigar.c_str(), &end, &a_cigar, &m));
    std::string converted_str = utils::cigar2str(n_cigar, a_cigar);
    CATCH_CHECK(cigar == converted_str);

    if (a_cigar) {
        hts_free(a_cigar);
    }
}

CATCH_TEST_CASE("BamUtilsTest: Test trim CIGAR", TEST_GROUP) {
    const std::string cigar = "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S";
    size_t m = 0;
    uint32_t *a_cigar = NULL;
    char *end = NULL;
    int n_cigar = int(sam_parse_cigar(cigar.c_str(), &end, &a_cigar, &m));
    const uint32_t qlen = uint32_t(bam_cigar2qlen(n_cigar, a_cigar));

    CATCH_SECTION("Trim nothing") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    CATCH_SECTION("Trim from first op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {1, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "11S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    CATCH_SECTION("Trim entire first op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {12, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "17M1D296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    CATCH_SECTION("Trim several ops from the front") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {29, qlen});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "296M2D21M1D3M2D10M1I320M1D2237M41S");
    }

    CATCH_SECTION("Trim from last op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen - 20});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M21S");
    }

    CATCH_SECTION("Trim entire last op") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen - 41});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M");
    }

    CATCH_SECTION("Trim several ops from the end") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {0, qlen - 2278});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "12S17M1D296M2D21M1D3M2D10M1I320M");
    }

    CATCH_SECTION("Trim from the middle") {
        auto ops = utils::trim_cigar(n_cigar, a_cigar, {29, qlen - 2278});
        std::string converted_str = utils::cigar2str(uint32_t(ops.size()), ops.data());
        CATCH_CHECK(converted_str == "296M2D21M1D3M2D10M1I320M");
    }

    if (a_cigar) {
        hts_free(a_cigar);
    }
}

CATCH_TEST_CASE("BamUtilsTest: Ref positions consumed", TEST_GROUP) {
    const std::string cigar = "12S17M1D296M2D21M1D3M2D10M1I320M1D2237M41S";
    size_t m = 0;
    uint32_t *a_cigar = NULL;
    char *end = NULL;
    int n_cigar = int(sam_parse_cigar(cigar.c_str(), &end, &a_cigar, &m));

    CATCH_SECTION("No positions consumed") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 0);
        CATCH_CHECK(pos_consumed == 0);
    }

    CATCH_SECTION("No positions consumed with soft clipping") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 12);
        CATCH_CHECK(pos_consumed == 0);
    }

    CATCH_SECTION("Match positions consumed") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 25);
        CATCH_CHECK(pos_consumed == 13);
    }

    CATCH_SECTION("Match and delete positions consumed") {
        auto pos_consumed = utils::ref_pos_consumed(n_cigar, a_cigar, 29);
        CATCH_CHECK(pos_consumed == 18);
    }

    if (a_cigar) {
        hts_free(a_cigar);
    }
}

CATCH_TEST_CASE("BamUtilsTest: Remove all alignment tags", TEST_GROUP) {
    fs::path bam_utils_test_dir = fs::path(get_data_dir("bam_utils"));
    auto sam = bam_utils_test_dir / "aligned_record.bam";

    HtsReader reader(sam.string(), std::nullopt);
    reader.set_add_filename_tag(false);
    CATCH_REQUIRE(reader.read());  // Parse first and only record.
    auto record = reader.record.get();

    utils::remove_alignment_tags_from_record(record);

    CATCH_CHECK(bam_aux_first(record) == nullptr);
}
