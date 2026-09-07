#include "TestUtils.h"
#include "secondary/common/bam_file.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace dorado::secondary::tests {

#define TEST_GROUP "[SecondaryBamFile]"

CATCH_TEST_CASE("Iterate  over all BAM records", TEST_GROUP) {
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_fn = test_data_dir / "in.aln.bam";

    // clang-format off
    const std::vector<std::string> expected_alns{
        "e0af6c87-8655-4603-97b7-0ad5ba860df2 26496 8621 18636  chr20 10000 0 10000 60 fl:i:0",
        "61ab09d6-072f-4ab2-b14b-b0a1e38a3419 23925 12152 22134  chr20 10000 0 10000 60 fl:i:0",
        "7d23577c-5c93-4d41-83bd-b652e687deee 18558 8035 18029  chr20 10000 0 10000 60 fl:i:0",
        "02551418-20c9-4b4b-9d1b-9bee36342895 17361 15465 17355  chr20 10000 0 1897 60 fl:i:0",
        "de45db56-e704-4524-af88-06a2f98c270e 19989 8179 18159 - chr20 10000 0 10000 60 fl:i:16",
        "c488f4c5-1639-4be1-92f6-948f29b7d822 35976 33532 35976 - chr20 10000 0 2452 60 fl:i:16",
        "563ecca1-30dd-4dd9-991a-d417d827c803 25424 14786 24762 - chr20 10000 0 10000 60 fl:i:16",
        "1e70cda3-c41f-4d19-9c14-94d8d64e619c 25824 12268 22265  chr20 10000 0 10000 60 fl:i:0",
        "b4139858-e420-4780-94e6-375542c2d2e8 25870 13311 23285  chr20 10000 0 10000 60 fl:i:0",
        "49b05d0d-97ac-449e-804b-35b35e05ce28 26284 8586 18455 - chr20 10000 0 10000 60 fl:i:16",
        "3fdc1b9b-7186-411e-af92-e93a1086754c 33926 13006 22983 - chr20 10000 0 10000 60 fl:i:16",
        "e7e27cb5-1144-49dd-8ec4-09a75937a091 40903 30191 40178 - chr20 10000 0 10000 60 fl:i:16",
        "4fd81aa2-cb77-4994-a8a5-70e6228f255e 36467 7797 17791  chr20 10000 0 10000 60 fl:i:0",
        "7b2095d4-08f7-448d-aa9d-55c9568fb49d 26134 20980 26126  chr20 10000 0 5143 60 fl:i:0",
        "dbe9785a-fa25-454c-9960-fd65fb99a040 40985 12546 22495 - chr20 10000 0 10000 60 fl:i:16",
        "a27cad27-2297-40d4-8666-40a4742eb2ed 41099 12569 22557  chr20 10000 0 10000 60 fl:i:0",
        "627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6 45332 21106 31089  chr20 10000 0 10000 60 fl:i:0",
        "3d7a9813-67be-4b84-b66a-0269aa108340 40938 20371 30366  chr20 10000 0 10000 60 fl:i:0",
        "ac863a7d-932e-42fa-91c1-7814d7f810f9 23736 17500 23725  chr20 10000 0 6234 60 fl:i:0",
        "d5560893-59c8-417c-a929-d62b4d19a1ca 28015 17332 23856  chr20 10000 0 6518 60 fl:i:0",
    };
    // clang-format on

    const std::vector<int64_t> expected_dwell_strides = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                                         6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    const std::vector<int64_t> expected_dwell_lens = {
            56754, 47065, 42781, 35940, 42893, 63531, 52216, 53638, 49746, 42344,
            74762, 91630, 75239, 53663, 76549, 83063, 81054, 82589, 45067, 70026};
    const std::vector<int64_t> expected_cigar_lens = {99, 103, 83, 47, 79,  34,  115, 67, 101, 327,
                                                      79, 73,  75, 43, 177, 177, 107, 91, 87,  117};

    std::vector<std::string> results_alns;
    std::vector<int64_t> results_dwell_strides;
    std::vector<int64_t> results_dwell_lens;
    std::vector<int64_t> results_cigar_lens;

    secondary::BamFile reader(in_fn, 1);

    while (BamPtr rec = reader.get_next()) {
        // Convert the alignment object.
        const secondary::Alignment aln = secondary::convert_bam1_to_aln(rec.get(), reader.hdr());

        results_alns.emplace_back(secondary::serialize_alignment_to_string(aln, " ", false));
        results_dwell_strides.emplace_back(aln.dwell_stride);
        results_dwell_lens.emplace_back(std::ssize(aln.dwells));
        results_cigar_lens.emplace_back(std::ssize(aln.cigar));
    }

    CATCH_CHECK(results_alns == expected_alns);
    CATCH_CHECK(results_dwell_strides == results_dwell_strides);
    CATCH_CHECK(results_dwell_lens == results_dwell_lens);
    CATCH_CHECK(results_cigar_lens == results_cigar_lens);
}

CATCH_TEST_CASE("get_view", TEST_GROUP) {
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_fn = test_data_dir / "in.aln.bam";

    secondary::BamFile reader(in_fn, 1);

    const secondary::BamFileView bfv = reader.get_view();

    CATCH_CHECK(bfv.fp == reader.fp());
    CATCH_CHECK(bfv.idx == reader.idx());
    CATCH_CHECK(bfv.hdr == reader.hdr());
}

CATCH_TEST_CASE("Fetch and iterate over all BAM records in a region", TEST_GROUP) {
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_fn = test_data_dir / "in.aln.bam";

    // clang-format off
    const std::vector<std::string> expected_alns{
        "e0af6c87-8655-4603-97b7-0ad5ba860df2 26496 8621 18636  chr20 10000 0 10000 60 fl:i:0",
        "61ab09d6-072f-4ab2-b14b-b0a1e38a3419 23925 12152 22134  chr20 10000 0 10000 60 fl:i:0",
        "7d23577c-5c93-4d41-83bd-b652e687deee 18558 8035 18029  chr20 10000 0 10000 60 fl:i:0",
        "02551418-20c9-4b4b-9d1b-9bee36342895 17361 15465 17355  chr20 10000 0 1897 60 fl:i:0",
        "de45db56-e704-4524-af88-06a2f98c270e 19989 8179 18159 - chr20 10000 0 10000 60 fl:i:16",
        "c488f4c5-1639-4be1-92f6-948f29b7d822 35976 33532 35976 - chr20 10000 0 2452 60 fl:i:16",
        "563ecca1-30dd-4dd9-991a-d417d827c803 25424 14786 24762 - chr20 10000 0 10000 60 fl:i:16",
        "1e70cda3-c41f-4d19-9c14-94d8d64e619c 25824 12268 22265  chr20 10000 0 10000 60 fl:i:0",
        "b4139858-e420-4780-94e6-375542c2d2e8 25870 13311 23285  chr20 10000 0 10000 60 fl:i:0",
        "49b05d0d-97ac-449e-804b-35b35e05ce28 26284 8586 18455 - chr20 10000 0 10000 60 fl:i:16",
        "3fdc1b9b-7186-411e-af92-e93a1086754c 33926 13006 22983 - chr20 10000 0 10000 60 fl:i:16",
        "e7e27cb5-1144-49dd-8ec4-09a75937a091 40903 30191 40178 - chr20 10000 0 10000 60 fl:i:16",
        "4fd81aa2-cb77-4994-a8a5-70e6228f255e 36467 7797 17791  chr20 10000 0 10000 60 fl:i:0",
        "7b2095d4-08f7-448d-aa9d-55c9568fb49d 26134 20980 26126  chr20 10000 0 5143 60 fl:i:0",
        "dbe9785a-fa25-454c-9960-fd65fb99a040 40985 12546 22495 - chr20 10000 0 10000 60 fl:i:16",
        "a27cad27-2297-40d4-8666-40a4742eb2ed 41099 12569 22557  chr20 10000 0 10000 60 fl:i:0",
        "627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6 45332 21106 31089  chr20 10000 0 10000 60 fl:i:0",
        "3d7a9813-67be-4b84-b66a-0269aa108340 40938 20371 30366  chr20 10000 0 10000 60 fl:i:0",
        "ac863a7d-932e-42fa-91c1-7814d7f810f9 23736 17500 23725  chr20 10000 0 6234 60 fl:i:0",
        "d5560893-59c8-417c-a929-d62b4d19a1ca 28015 17332 23856  chr20 10000 0 6518 60 fl:i:0",
    };
    // clang-format on

    const std::vector<int64_t> expected_dwell_strides = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                                         6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
    const std::vector<int64_t> expected_dwell_lens = {
            56754, 47065, 42781, 35940, 42893, 63531, 52216, 53638, 49746, 42344,
            74762, 91630, 75239, 53663, 76549, 83063, 81054, 82589, 45067, 70026};
    const std::vector<int64_t> expected_cigar_lens = {99, 103, 83, 47, 79,  34,  115, 67, 101, 327,
                                                      79, 73,  75, 43, 177, 177, 107, 91, 87,  117};

    std::vector<std::string> results_alns;
    std::vector<int64_t> results_dwell_strides;
    std::vector<int64_t> results_dwell_lens;
    std::vector<int64_t> results_cigar_lens;

    secondary::BamFile reader(in_fn, 1);

    auto it = reader.fetch("chr20", 100, 2000);

    while (BamPtr rec = it.get_next()) {
        // Convert the alignment object.
        const secondary::Alignment aln = secondary::convert_bam1_to_aln(rec.get(), reader.hdr());

        results_alns.emplace_back(secondary::serialize_alignment_to_string(aln, " ", false));
        results_dwell_strides.emplace_back(aln.dwell_stride);
        results_dwell_lens.emplace_back(std::ssize(aln.dwells));
        results_cigar_lens.emplace_back(std::ssize(aln.cigar));
    }

    CATCH_CHECK(results_alns == expected_alns);
    CATCH_CHECK(results_dwell_strides == results_dwell_strides);
    CATCH_CHECK(results_dwell_lens == results_dwell_lens);
    CATCH_CHECK(results_cigar_lens == results_cigar_lens);
}

CATCH_TEST_CASE("Fetch region out of bounds of the reference should return empty", TEST_GROUP) {
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_fn = test_data_dir / "in.aln.bam";

    const std::vector<std::string> expected{};

    secondary::BamFile reader(in_fn, 1);

    auto it = reader.fetch("chr20", 100000, 100100);

    std::vector<std::string> results;

    while (BamPtr rec = it.get_next()) {
        // Convert the alignment object.
        const secondary::Alignment aln = secondary::convert_bam1_to_aln(rec.get(), reader.hdr());

        // Serialize the BAM record for comparison.
        results.emplace_back(secondary::serialize_alignment_to_string(aln, " ", false));
    }

    CATCH_CHECK(results == expected);
}

CATCH_TEST_CASE("Fetch region with negative start coord should return empty", TEST_GROUP) {
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_fn = test_data_dir / "in.aln.bam";

    const std::vector<std::string> expected{};

    secondary::BamFile reader(in_fn, 1);

    auto it = reader.fetch("chr20", -100000, 0);

    std::vector<std::string> results;

    while (BamPtr rec = it.get_next()) {
        // Convert the alignment object.
        const secondary::Alignment aln = secondary::convert_bam1_to_aln(rec.get(), reader.hdr());

        // Serialize the BAM record for comparison.
        results.emplace_back(secondary::serialize_alignment_to_string(aln, " ", false));
    }

    CATCH_CHECK(results == expected);
}

CATCH_TEST_CASE(
        "Fetch region with negative end coord should throw because region string is ambiguous",
        TEST_GROUP) {
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_fn = test_data_dir / "in.aln.bam";

    const std::vector<std::string> expected{};

    secondary::BamFile reader(in_fn, 1);

    CATCH_CHECK_THROWS(reader.fetch("chr20", -100000, -5));
}

}  // namespace dorado::secondary::tests