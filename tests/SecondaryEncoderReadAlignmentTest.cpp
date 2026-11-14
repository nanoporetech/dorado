#include "../dorado/secondary/features/encoder_read_alignment.h"
#include "TestUtils.h"
#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "secondary/features/encoder_utils.h"
#include "secondary/features/haplotag_source.h"
#include "secondary/features/kadayashi_options.h"
#include "utils/cigar.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <htslib/sam.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define TEST_GROUP "[SecondaryEncoderReadAlignment]"

namespace {

// Utility struct to make test cases.
struct BamRecord {
    std::string_view qname{};
    int32_t tid{0};
    int32_t pos{0};
    uint16_t flag{0};
    uint8_t mapq{0};
    std::string_view cigar{};
    std::string md{};
    std::string_view seq{};
    std::string_view qual{};
    std::vector<int8_t> dwell_tag{};
    std::optional<int32_t> hp_tag{};
    std::optional<int32_t> nm{0};
};

// Utility to produce a synthetic BAM.
// Parses a CIGAR from string, but returns a HTS-style vector of packed lengths and ops.
std::vector<uint32_t> parse_cigar_from_string_hts(const std::string_view cigar) {
    std::vector<dorado::CigarOp> parsed = dorado::parse_cigar_from_string(cigar);
    std::vector<uint32_t> ret(std::size(parsed));
    for (size_t i = 0; i < std::size(parsed); ++i) {
        const dorado::CigarOp cig = parsed[i];
        const int8_t op = static_cast<int8_t>(cig.op);
        ret[i] = bam_cigar_gen(cig.len, op);
    }
    return ret;
}

// Utility to produce a synthetic BAM.
// Construct a BAM header given a vector of (name, length) pairs
dorado::SamHdrPtr make_bam_hdr(const std::span<const std::pair<std::string, std::string>> seqs) {
    // Create the header as text.
    std::string text = "@HD\tVN:1.6\tSO:unknown\n";
    for (const auto& [name, seq] : seqs) {
        text += "@SQ\tSN:" + name + "\tLN:" + std::to_string(std::size(seq)) + "\n";
    }

    // Parse it.
    dorado::SamHdrPtr hdr{sam_hdr_parse(std::size(text), text.c_str())};

    return hdr;
}

// Utility to produce a synthetic BAM.
// Constructs a HTS-style BAM record (bam1_t*) from given data.
dorado::BamPtr make_bam1(const BamRecord& record) {
    const std::vector<uint32_t> cigar_vec = parse_cigar_from_string_hts(record.cigar);
    if (std::empty(record.md)) {
        throw std::runtime_error{"Synthetic BAM record is missing an explicit MD tag."};
    }

    dorado::BamPtr ret{bam_init1()};

    bam_set1(ret.get(), std::size(record.qname), std::data(record.qname), record.flag, record.tid,
             record.pos, record.mapq, std::size(cigar_vec), std::data(cigar_vec), -1, -1, 0,
             std::size(record.seq), std::data(record.seq),
             std::empty(record.qual) ? nullptr : std::data(record.qual), 0);

    // Move table specification: https://software-docs.nanoporetech.com/dorado/latest/basecaller/move_table
    // Tag format: `mv:B:c,[block_stride],[signal_block_move_list]
    if (!std::empty(record.dwell_tag)) {
        std::vector<uint8_t> aux_data;
        aux_data.emplace_back('c');  // subtype: int8_t

        // Number of elements. The +1 is because of the block_stride which will be added below.
        const uint32_t n = static_cast<uint32_t>(std::size(record.dwell_tag));
        const uint8_t* n_bytes = reinterpret_cast<const uint8_t*>(&n);
        aux_data.insert(std::end(aux_data), n_bytes, n_bytes + sizeof(n));

        // Add each int8_t value
        for (const int8_t val : record.dwell_tag) {
            aux_data.emplace_back(static_cast<uint8_t>(val));
        }

        // Add the move table.
        bam_aux_append(ret.get(), "mv", 'B', std::size(aux_data), std::data(aux_data));
    }

    // Add HP:i tag
    if (record.hp_tag) {
        bam_aux_append(ret.get(), "HP", 'i', sizeof(int32_t),
                       reinterpret_cast<const uint8_t*>(&(*record.hp_tag)));
    }

    // Add the edit distance tag.
    if (record.nm) {
        bam_aux_append(ret.get(), "NM", 'i', sizeof(int32_t),
                       reinterpret_cast<const uint8_t*>(&(*record.nm)));
    }

    const int32_t md_len = static_cast<int32_t>(std::size(record.md) + 1);
    bam_aux_append(ret.get(), "MD", 'Z', md_len,
                   reinterpret_cast<const uint8_t*>(record.md.c_str()));

    return ret;
}

// Utility to produce a synthetic BAM.
// Writes a BAM file given BAM records and reference sequences.
void write_bam(const std::filesystem::path& out_fn,
               const std::span<const std::pair<std::string, std::string>> targets,
               const std::span<const BamRecord> records) {
    dorado::utils::HtsFile hts_file(out_fn.string(), dorado::utils::HtsFile::OutputMode::BAM, 1,
                                    true);
    dorado::SamHdrPtr header = make_bam_hdr(targets);
    hts_file.set_header(header.get());
    for (const BamRecord& r : records) {
        if ((r.tid < 0) || (r.tid >= std::ssize(targets))) {
            throw std::runtime_error{"Synthetic BAM record has an invalid target id."};
        }
        dorado::BamPtr bam_record = make_bam1(r);
        hts_file.write(bam_record.get());
    }
    hts_file.finalise([](float) { /* noop */ });
}

// Utility to write reference sequences to a FASTA file.
void write_ref(const std::filesystem::path& out_fn,
               const std::span<const std::pair<std::string, std::string>> targets) {
    std::ofstream ofs(out_fn);
    for (const auto& [name, seq] : targets) {
        ofs << ">" << name << "\n" << seq << "\n";
    }
}

void eval_sample(const dorado::secondary::Sample& expected,
                 const dorado::secondary::Sample& result) {
    CATCH_CHECK(result.seq_id == expected.seq_id);
    CATCH_CHECK(result.read_ids_left == expected.read_ids_left);
    CATCH_CHECK(result.read_ids_right == expected.read_ids_right);
    CATCH_CHECK(result.positions_major == expected.positions_major);
    CATCH_CHECK(result.positions_minor == expected.positions_minor);
    CATCH_CHECK(result.depth.equal(expected.depth));
    CATCH_CHECK(result.features.equal(expected.features));
}

template <typename T>
std::vector<T> get_unique_sorted_column(const at::Tensor& in_tensor, const int64_t column_id) {
    using namespace torch::indexing;
    const torch::Tensor flat =
            in_tensor.index({Slice(), Slice(), column_id}).flatten().to(torch::kCPU).contiguous();
    const T* data = flat.data_ptr<T>();
    const int64_t n = flat.numel();
    std::unordered_set<T> uniq;
    uniq.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        uniq.insert(data[i]);
    }
    std::vector<T> uniq_vec(std::begin(uniq), std::end(uniq));
    std::sort(std::begin(uniq_vec), std::end(uniq_vec));
    return uniq_vec;
}

struct EncoderTestCase {
    std::string test_name;
    bool legacy_feat_encoder;
};

}  // namespace

namespace dorado::secondary::tests {
CATCH_TEST_CASE("read_ids", TEST_GROUP) {
    /*
        The selected window covers full alignments, none of them spanning past the end position.

        The input BAM consists of the following alignments (truncated to not show CIGARs for brewity):
            6d44bf1b-f611-4663-9ee2-d655675471af    4637    3355    4637    +       contig_1        10000   0       1292    1279    1282    60
            e4c37c19-8cfe-49a8-bb4e-391463093536    10664   10385   10652   +       contig_1        10000   0       270     266     267     60
            8c9df1f1-513e-4756-8259-3d541bb92b02    23233   3004    12981   +       contig_1        10000   0       10000   9949    9977    60
            be8030bd-79f6-4b77-b45a-79b3a9bf2fc4    29700   16165   25976   +       contig_1        10000   0       10000   9719    9811    60
            bdfeee16-390f-43a7-938b-9502ce984921    29158   11920   21922   +       contig_1        10000   0       10000   9954    10002   60
            9ab794fd-8b68-40d3-b0da-a2c52634d2c0    48570   36461   46453   +       contig_1        10000   0       10000   9939    9992    60
            6698108e-3855-4315-a01d-429f3a74e2a8    1833    0       1812    -       contig_1        10000   0       1808    1805    1812    60
            bf84ebf6-5820-4b4d-bb17-e16ecd14b856    26003   0       3126    -       contig_1        10000   0       3141    3118    3126    60
            6a3a4f34-c012-457c-9c82-5e384caa6bd0    28767   8226    18204   -       contig_1        10000   0       10000   9954    9978    60
            daf9f60a-8d26-4b84-ae41-dc40a47b5e8e    22452   2497    12478   -       contig_1        10000   0       10000   9936    9981    60
            8c06fcee-fe52-4c3b-be39-ce92a87e5d31    43433   12739   22740   -       contig_1        10000   2       10000   9949    10001   60
            5b4a36c3-92f1-45ca-81d7-9440e4a7031c    22810   0       8796    +       contig_1        10000   1201    10000   8765    8796    60
            e3d50dff-b679-4404-89ed-3a17ec47a146    1017    35      1017    -       contig_1        10000   1621    2598    974     982     60
            fc40fcd9-5967-4cce-8dcd-b11675508925    13419   5971    13419   -       contig_1        10000   2543    10000   7398    7448    60
            6db40560-2461-41f4-a221-d855ef17f6b1    2184    0       2184    +       contig_1        10000   2888    5079    2182    2184    60
            1def391b-dd69-4b42-99cb-c5a899de0d0a    4300    0       4290    +       contig_1        10000   4785    9091    4274    4290    60
            667f963f-68c1-40c8-9af3-96190eca411f    39551   38529   39542   -       contig_1        10000   8985    10000   1009    1013    60

        When `row_per_read == false`, the encoder will try to add new alignments to the rows which freed up.
        In this case, these are the orders of alignments per row which are expected:
            Row Coordinates/qnames
            0   0-1292 (6d44bf1b-f611-4663-9ee2-d655675471af); (12) 1621-2598 (e3d50dff-b679-4404-89ed-3a17ec47a146), (14) 2888-5079 (6db40560-2461-41f4-a221-d855ef17f6b1), (16) 8985-10000 (667f963f-68c1-40c8-9af3-96190eca411f)
            1   0-270 (e4c37c19-8cfe-49a8-bb4e-391463093536), (11) 1201-10000 (5b4a36c3-92f1-45ca-81d7-9440e4a7031c)
            2   0-10000 (8c9df1f1-513e-4756-8259-3d541bb92b02)
            3   0-10000 (be8030bd-79f6-4b77-b45a-79b3a9bf2fc4)
            4   0-10000 (bdfeee16-390f-43a7-938b-9502ce984921)
            5   0-10000 (9ab794fd-8b68-40d3-b0da-a2c52634d2c0)
            6   0-1808 (6698108e-3855-4315-a01d-429f3a74e2a8), (13) 2543-10000 (fc40fcd9-5967-4cce-8dcd-b11675508925)
            7   0-3141 (bf84ebf6-5820-4b4d-bb17-e16ecd14b856), (15) 4785-9091 (1def391b-dd69-4b42-99cb-c5a899de0d0a)
            8   0-10000 (6a3a4f34-c012-457c-9c82-5e384caa6bd0)
            9   0-10000 (daf9f60a-8d26-4b84-ae41-dc40a47b5e8e)
            10  2-10000 (8c06fcee-fe52-4c3b-be39-ce92a87e5d31)

        So all but the last row (row ID 10) should have the read_ids_left set to a qname, and all but the row ID 7 should
        have the read_ids_right set to a qname.
    */

    // Test data.
    const std::filesystem::path test_data_dir = get_data_dir("polish") / "test-01-supertiny";
    const std::filesystem::path in_ref_fn{test_data_dir / "draft.fasta.gz"};
    const std::filesystem::path in_bam_aln_fn{test_data_dir / "calls_to_draft.bam"};

    // Expected results.
    // clang-format off
    const std::vector<std::string> expected_read_ids_left{
        "6d44bf1b-f611-4663-9ee2-d655675471af",
        "e4c37c19-8cfe-49a8-bb4e-391463093536",
        "8c9df1f1-513e-4756-8259-3d541bb92b02",
        "be8030bd-79f6-4b77-b45a-79b3a9bf2fc4",
        "bdfeee16-390f-43a7-938b-9502ce984921",
        "9ab794fd-8b68-40d3-b0da-a2c52634d2c0",
        "6698108e-3855-4315-a01d-429f3a74e2a8",
        "bf84ebf6-5820-4b4d-bb17-e16ecd14b856",
        "6a3a4f34-c012-457c-9c82-5e384caa6bd0",
        "daf9f60a-8d26-4b84-ae41-dc40a47b5e8e",
        "__blank",
    };
    const std::vector<std::string> expected_read_ids_right{
        "667f963f-68c1-40c8-9af3-96190eca411f",
        "5b4a36c3-92f1-45ca-81d7-9440e4a7031c",
        "8c9df1f1-513e-4756-8259-3d541bb92b02",
        "be8030bd-79f6-4b77-b45a-79b3a9bf2fc4",
        "bdfeee16-390f-43a7-938b-9502ce984921",
        "9ab794fd-8b68-40d3-b0da-a2c52634d2c0",
        "fc40fcd9-5967-4cce-8dcd-b11675508925",
        "__blank",
        "6a3a4f34-c012-457c-9c82-5e384caa6bd0",
        "daf9f60a-8d26-4b84-ae41-dc40a47b5e8e",
        "8c06fcee-fe52-4c3b-be39-ce92a87e5d31",
    };
    // clang-format on

    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const bool include_dwells{true};
    const bool clip_to_zero{true};
    const bool right_align_insertions{false};
    const bool include_haplotype_column{false};
    const bool include_snp_qv_column{false};
    const double min_snp_accuracy{0.0};
    const HaplotagSource hap_source{HaplotagSource::UNPHASED};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;

    CATCH_SECTION("Region has no overhangs") {
        /**
         * The region here is `contig_1:1-10000` which means that the window ends at the same position where all alignments end.
         * This means that the `bam_mplp_auto` function will exit the while loop instead of it being broken by an internal
         * condition.
         * When this happens, the `pos` value is set to 0 instead of 10000 (that's what `bam_mplp_auto` does).
         *
         * There was a bug in the `calculate_read_alignment()` function which compared `pos` with each read_array element's
         * end position to mark the `read_ids_right`; because `pos` was 0, it would mark all of the reads in the read_array
         * as reaching the end of the contig. But if any of the reads spanned past the end of the region, then
         * the while loop would have been terminated before and the tagging of left/right IDs would work fine.
         *
         * The last position (last_pos) is now tracked internally to fix this.
        */

        const int64_t ref_end = 10000;

        const std::vector<int64_t> expected_shape{10432, 11, 5};

        // UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                in_ref_fn, in_bam_aln_fn, dtypes, tag_name, tag_value, tag_keep_missing, read_group,
                min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells, clip_to_zero,
                right_align_insertions, include_haplotype_column, hap_source, phasing_bin,
                include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        const std::vector<int64_t> shape(std::cbegin(result.features.sizes()),
                                         std::cend(result.features.sizes()));

        CATCH_CHECK(result.read_ids_left == expected_read_ids_left);
        CATCH_CHECK(result.read_ids_right == expected_read_ids_right);
        CATCH_CHECK(shape == expected_shape);
    }

    CATCH_SECTION("With an overhang of 1bp on the right end of the region") {
        /**
         * Unlike the previous test, the window here is 1bp shorter which means that most rows will be clipped
         * by 1bp. This is intended to test the `read_ids_left` and `read_ids_right` which had a bug.
         * (The bug was that labeling on the right end was performed if read.ref_end >= pos, but if
         * the bam_mplp_auto reached the end of the window it will set `pos == 0` instead to that position).
        */

        const int64_t ref_end = 9999;

        const std::vector<int64_t> expected_shape{10431, 11, 5};

        // UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                in_ref_fn, in_bam_aln_fn, dtypes, tag_name, tag_value, tag_keep_missing, read_group,
                min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells, clip_to_zero,
                right_align_insertions, include_haplotype_column, hap_source, phasing_bin,
                include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        const std::vector<int64_t> shape(std::begin(result.features.sizes()),
                                         std::end(result.features.sizes()));

        CATCH_CHECK(result.read_ids_left == expected_read_ids_left);
        CATCH_CHECK(result.read_ids_right == expected_read_ids_right);
        CATCH_CHECK(shape == expected_shape);
    }
}

CATCH_TEST_CASE("Compute haptags", TEST_GROUP) {
    /**
     * This test computes the haptags internally instead of loading them from the BAM file.
     *
     * NOTE: Kadayashi is actually tested in another unit test file. This only checks the code path which
     * invokes Kadayashi. The difference is that the internally hardcoded parameters of the medaka_read_matrix
     * are too stringent for this small test case and all haptags end up equal to zero here.
     * However, this tests provides code coverage for this path.
     */

    // Test data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_bam_aln_fn = test_data_dir / "in.aln.bam";
    const std::filesystem::path in_ref_fn = test_data_dir / "in.ref.fasta.gz";

    // Parameters which we aren't testing at the moment.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const bool clip_to_zero{true};
    const bool right_align_insertions{false};
    const bool include_dwells{true};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{false};
    const double min_snp_accuracy{0.0};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"chr20"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 10000;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;

    // Important for this test - testing that the COMPUTE path works.
    const HaplotagSource hap_source{HaplotagSource::COMPUTE};

    // Expected results.
    const std::vector<int64_t> expected_shape{10491, 20, 6};
    const std::vector<int8_t> expected_haptags{0};

    // Run UUT.
    auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
            EncoderTestCase{"kadayashi feature encoder", false},
            EncoderTestCase{"legacy feature encoder", true},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
    EncoderReadAlignment encoder(
            in_ref_fn, in_bam_aln_fn, dtypes, tag_name, tag_value, tag_keep_missing, read_group,
            min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells, clip_to_zero,
            right_align_insertions, include_haplotype_column, hap_source, phasing_bin,
            include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);

    const Sample result = encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

    const std::vector<int64_t> shape(std::begin(result.features.sizes()),
                                     std::end(result.features.sizes()));

    const std::vector<int8_t> haptags = get_unique_sorted_column<int8_t>(result.features, -1);

    CATCH_CHECK(shape == expected_shape);
    CATCH_CHECK(haptags == expected_haptags);
}

CATCH_TEST_CASE("snp_accuracy_filter", TEST_GROUP) {
    /**
     * This is a list of all reads in the input BAM, with alignment accuracy (second column) and SNP accuracy (third column):
     *       qname                                   aln_accc    snp_acc
     *       6d44bf1b-f611-4663-9ee2-d655675471af	0.982239	0.994527
     *       e4c37c19-8cfe-49a8-bb4e-391463093536	0.9631	    0.981203
     *       8c9df1f1-513e-4756-8259-3d541bb92b02	0.988133	0.995979
     *       be8030bd-79f6-4b77-b45a-79b3a9bf2fc4	0.944411	0.980656
     *       bdfeee16-390f-43a7-938b-9502ce984921	0.987659	0.996986
     *       9ab794fd-8b68-40d3-b0da-a2c52634d2c0	0.986173	0.997485
     *       6698108e-3855-4315-a01d-429f3a74e2a8	0.992287	0.997784
     *       bf84ebf6-5820-4b4d-bb17-e16ecd14b856	0.984757	0.994548
     *       6a3a4f34-c012-457c-9c82-5e384caa6bd0	0.989425	0.996383
     *       daf9f60a-8d26-4b84-ae41-dc40a47b5e8e	0.984072	0.994867
     *       8c06fcee-fe52-4c3b-be39-ce92a87e5d31	0.986667	0.996683
     */
    // Test data.
    const std::filesystem::path test_data_dir = get_data_dir("polish") / "test-01-supertiny";
    const std::filesystem::path in_ref_fn{test_data_dir / "draft.fasta.gz"};
    const std::filesystem::path in_bam_aln_fn{test_data_dir / "calls_to_draft.bam"};

    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const bool include_dwells{true};
    const bool clip_to_zero{true};
    const bool right_align_insertions{false};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{false};
    const HaplotagSource hap_source{HaplotagSource::BAM_HAP_TAG};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 111;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;

    struct TestCase {
        std::string test_name;
        double min_snp_accuracy{0.0};
        std::vector<int64_t> expected_shape{};
    };

    auto [encoder_test_case] = GENERATE_REF(table<EncoderTestCase>({
            EncoderTestCase{"kadayashi feature encoder", false},
            EncoderTestCase{"legacy feature encoder", true},
    }));

    auto [test_case] = GENERATE_REF(table<TestCase>({
            TestCase{"No filter", 0.0, {117, 11, 6}},
            TestCase{"Filter some alignments", 0.99, {113, 9, 6}},
            TestCase{"Filter ALL alignments", 1.0, {0}},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name << ", "
                          << encoder_test_case.test_name);

    // Run UUT.
    EncoderReadAlignment encoder(
            in_ref_fn, in_bam_aln_fn, dtypes, tag_name, tag_value, tag_keep_missing, read_group,
            min_mapq, max_reads, test_case.min_snp_accuracy, row_per_read, include_dwells,
            clip_to_zero, right_align_insertions, include_haplotype_column, hap_source, phasing_bin,
            include_snp_qv_column, kadayashi_opt, encoder_test_case.legacy_feat_encoder);

    const Sample result = encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

    const std::vector<int64_t> shape(std::begin(result.features.sizes()),
                                     std::end(result.features.sizes()));

    CATCH_CHECK(shape == test_case.expected_shape);
}

CATCH_TEST_CASE("synthetic_test_01", TEST_GROUP) {
    /**
     * This tests the full output of the encode_region() function on a synthetic
     * test case constructed within the test.
     *
     * This tests:
     * - Reads with and without QVs.
     * - Forward/reverse strands.
     * - Left/right sequence IDs for reads which begin at the first base of the window
     *   or end at the very last base of the window. Other rows should be blank.
     * - Duplicate input reads - this exposed 2 bugs: (1) the max_n_reads was too large because it included duplicate
     *   read IDs, which resulted in uninitialized rows in read_ids_left and read_ids_right; (2) because of the uninitialized rows,
     *   reordering chunks and merging them was throwing an exception because the output tensor had more rows than it should have.
     * - Multiple non-overlapping reads which should be placed on the sam row in the read_array (and the tensor).
     * - Switch the non-base feature columns on/off (dwells, haplotags, snp_qv).
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGAACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 50 /*mapq*/, "10M" /*cigar*/, "10" /*MD*/, "ACTGAACTGA" /*seq*/, "" /*qual*/, {} /*dwell*/, {} /*hp*/, 0 /*NM*/},    // Full-span.
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 51 /*mapq*/, "10M", "10", "ACTGAACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 52 /*mapq*/, "10M", "10", "ACTGAACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 16 /*flag*/, 53 /*mapq*/, "5M", "0A4", "ACTGA", "", {}, 3, 1},                                    // Left-flank only. Reverse. NM = 1 and cigar = 5M make snp_qv = 6.98.
            {"read_03", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 54 /*mapq*/, "3M", "0A0C1", "ACT", "", {5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 5, 2},  // This should share the row with the next one. NM = 2 and cigar = 3M make snp_qv = 1.76.
            {"read_04", 0 /*tid*/, 8 /*pos*/, 0  /*flag*/, 55 /*mapq*/, "2M", "0G1", "GA", "", {}, {}, 1},                                      // This should share the row with the previous.
            {"read_05", 0 /*tid*/, 1 /*pos*/, 0  /*flag*/, 56 /*mapq*/, "8M", "0C0T0G5", "CTGAACTG", "1234567890", {}, {}, 3},                  // Contained. Has quals. NM = 3 and cigar = 8M make snp_qv = 4.26.
            {"read_06", 0 /*tid*/, 5 /*pos*/, 0  /*flag*/, 57 /*mapq*/, "5M", "0A0C0T0G0A0", "ACTGA", "", {}, {}, 5},                           // Right-flank only.
            {"read_06", 0 /*tid*/, 5 /*pos*/, 0  /*flag*/, 58 /*mapq*/, "5M", "5", "ACTGA", "", {}, {}, 0},                                     // Duplicate.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    const Sample expected_total {
        .seq_id = 123,
        // Features tensor shape: [pos, coverage, features] -> [10, 5, 7]
        // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
        .features = torch::tensor(
            {
                // (0,.,.)
                {{1, 0, 1, 50, 0, 0, 60},
                 {1, 0, 0, 53, 0, 3, 7},
                 {1, 0, 1, 54, 4, 5, 2},
                 {0, 0, 0,  0, 0, 0, 0},
                 {0, 0, 0,  0, 0, 0, 0}},

                // (1,.,.)
                {{2, 0,  1, 50, 0, 0, 60},
                 {2, 0,  0, 53, 0, 3,  7},
                 {2, 0,  1, 54, 5, 5,  2},
                 {2, 49, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (2,.,.)
                {{4, 0,  1, 50, 0, 0, 60},
                 {4, 0,  0, 53, 0, 3, 7},
                 {4, 0,  1, 54, 2, 5, 2},
                 {4, 50, 1, 56, 0, 0, 4},
                 {0, 0,  0,  0, 0, 0, 0}},

                // (3,.,.)
                {{3, 0,  1, 50, 0, 0, 60},
                 {3, 0,  0, 53, 0, 3,  7},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 51, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (4,.,.)
                {{1,  0,  1, 50, 0, 0, 60},
                 {1,  0,  0, 53, 0, 3,  7},
                 {0,  0,  0,  0, 0, 0,  0},
                 {1, 52,  1, 56, 0, 0,  4},
                 {0,  0,  0,  0, 0, 0,  0}},

                // (5,.,.)
                {{1, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {1, 53, 1, 56, 0, 0,  4},
                 {1, 0,  1, 57, 0, 0,  0}},

                // (6,.,.)
                {{2, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {2, 54, 1, 56, 0, 0,  4},
                 {2, 0,  1, 57, 0, 0,  0}},

                // (7,.,.)
                {{4, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {4, 55, 1, 56, 0, 0,  4},
                 {4, 0,  1, 57, 0, 0,  0}},

                // (8,.,.)
                {{3, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 0,  1, 55, 0, 0,  3},
                 {3, 56, 1, 56, 0, 0,  4},
                 {3, 0,  1, 57, 0, 0,  0}},

                // (9,.,.)
                {{1, 0, 1, 50, 0, 0, 60},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 0, 1, 55, 0, 0,  3},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 0, 1, 57, 0, 0,  0}},

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        },
        .positions_minor = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        .depth = torch::tensor(
            {3, 4, 4, 3, 3, 3, 3, 3, 4, 3},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
            "read_03",
            "__blank",
            "__blank",
        },
        .read_ids_right = {
            "read_01",
            "__blank",
            "read_04",
            "__blank",
            "read_06",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const double min_snp_accuracy{0.0};

    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 10;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;

    CATCH_SECTION(
            "No dwell column, no hap column, no snp_qv column. Only the base features (base, qual, "
            "strand, mapq)."
            "Tests the entire output Sample: feature tensor, depth tensor, positions major/minor, "
            "and read IDs. Include dwell col, no hap col") {
        // Test specific parameters.
        const bool include_dwells{false};
        const bool include_haplotype_column{false};
        const bool include_snp_qv_column{false};
        const HaplotagSource hap_source{HaplotagSource::UNPHASED};
        const bool clip_to_zero{true};

        // Expected results for this test: drop the dwell column from the last dimension.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index({"...", torch::indexing::Slice(0, 4)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }

    CATCH_SECTION("Only dwell column, no hap column and no snp_qv column.") {
        // Test specific parameters.
        const bool include_dwells{true};
        const bool include_haplotype_column{false};
        const bool include_snp_qv_column{false};
        const HaplotagSource hap_source{HaplotagSource::UNPHASED};
        const bool clip_to_zero{true};

        // Expected results for this test. Drop the haplotag column.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index({"...", torch::indexing::Slice(0, 5)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);

        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }

    CATCH_SECTION("Only haptag column, no dwell and no snp_qv columns.") {
        // Test specific parameters.
        const bool include_dwells{false};
        const bool include_haplotype_column{true};
        const bool include_snp_qv_column{false};
        const HaplotagSource hap_source{HaplotagSource::BAM_HAP_TAG};
        const bool clip_to_zero{true};

        // Expected results for this test: drop the dwell column from the last dimension.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index(
                        {torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::tensor({0, 1, 2, 3, 5}, torch::kLong)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }

    CATCH_SECTION("Only snp_qv column, no dwell and no hap columns.") {
        // Test specific parameters.
        const bool include_dwells{false};
        const bool include_haplotype_column{false};
        const bool include_snp_qv_column{true};
        const HaplotagSource hap_source{HaplotagSource::UNPHASED};
        const bool clip_to_zero{true};

        // Expected results for this test: drop the dwell column from the last dimension.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index(
                        {torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::tensor({0, 1, 2, 3, 6}, torch::kLong)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }

    CATCH_SECTION("Use dwell and snp_qv columns, but no haptag column.") {
        // Test specific parameters.
        const bool include_dwells{true};
        const bool include_haplotype_column{false};
        const bool include_snp_qv_column{true};
        const HaplotagSource hap_source{HaplotagSource::UNPHASED};
        const bool clip_to_zero{true};

        // Expected results for this test: drop the dwell column from the last dimension.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index(
                        {torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::tensor({0, 1, 2, 3, 4, 6}, torch::kLong)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }

    CATCH_SECTION(
            "Everything - with dwell column, hap column and snp_qv column. All else should be the "
            "same as before.") {
        // Test specific parameters.
        const bool include_dwells{true};
        const bool include_haplotype_column{true};
        const bool include_snp_qv_column{true};
        const HaplotagSource hap_source{HaplotagSource::BAM_HAP_TAG};
        const bool clip_to_zero{true};

        // Expected results for this test, including the dwell and the haplotag columns. add a column of zeros for the haplotype for each read.
        const Sample& expected = expected_total;

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);

        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }

    CATCH_SECTION(
            "Test clip_to_zero == false. In this case, the strand "
            "should be -1 or 1, instead of 0 and 1. Other values identical") {
        // Test specific parameters.
        const bool include_dwells{true};
        const bool include_haplotype_column{true};
        const bool include_snp_qv_column{true};
        const HaplotagSource hap_source{HaplotagSource::BAM_HAP_TAG};
        const bool clip_to_zero{false};

        // Expected results for this test.
        // Edit the expected_features_total so that the strand column (col2) has values {-1, 1} and
        // reads with no quals should have a -1 in the col1 column.
        Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.clone(),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };
        {
            // Grab slices.
            const at::Tensor col0 = expected.features.index({"...", 0});
            const at::Tensor col1 = expected.features.index({"...", 1});
            const at::Tensor col2 = expected.features.index({"...", 2});

            // Update col2 (strand) where col0 != 0 so that strand is either -1 or 1.
            const at::Tensor mask_col0_nonzero = (col0 != 0);
            const at::Tensor updated_col2 = col2 * 2 - 1;
            expected.features.index_put_({"...", 2},
                                         torch::where(mask_col0_nonzero, updated_col2, col2));

            // Update col1 (qual) where col0 != 0 && col1 == 0.
            // This is because HTSlib will return -1 for each base if the input
            // BAM record does not contain qualities.
            const at::Tensor mask_col0_and_col1 = (col0 != 0) & (col1 == 0);
            expected.features.index_put_(
                    {"...", 1}, torch::where(mask_col0_and_col1, torch::full_like(col1, -1), col1));
        }

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);

        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }
}

CATCH_TEST_CASE("synthetic_test_02", TEST_GROUP) {
    /**
     * Tests the insertion columns, deletion base values and multiple reads per row.
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        // Target (reference) sequences.
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGAACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "8M1D1M", "8^G1", "ACTGAACTA", "", {}, {}, 0},  // Full-span.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 0 /*flag*/,  60 /*mapq*/, "2M2I1M", "3", "ACAAT", "", {}, {}, 0},         // Reusable row, part 1.
            {"read_03", 0 /*tid*/, 8 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "2M", "2", "GA", "", {}, {}, 0},                // Reusable row, part 2.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    /**
     * Columns of the feature tensor are:
     *      base, qual, strand, mapq, dwell, haplotype [, dtype]
     *
     * NOTE: The base column is encoded in the following way:
     *      A = 1, C = 2, G = 3, T = 4, DEL = 5
     *
     *      The strand column is:
     *          FWD = 1, REV = 0
     */
    const Sample expected {
        .seq_id = 123,
        .features = torch::tensor(
            {
                // (0,.,.) "A"/"A", major/minor: 0/0
                {{1, 0, 1, 60, 0, 0},
                {1, 0, 1, 60, 0, 0}},

                // (1,.,.) "C"/"C", major/minor: 1/0
                {{2, 0, 1, 60, 0, 0},
                {2, 0, 1, 60, 0, 0}},

                // (2,.,.)  -> Insertion ("-"/"A"), major/minor: 1/1
                {{5, 0, 1, 60, 0, 0},
                {1, 0, 1, 60, 0, 0}},

                // (3,.,.)  -> Insertion ("-"/"A"), major/minor: 1/2
                {{5, 0, 1, 60, 0, 0},
                {1, 0, 1, 60, 0, 0}},

                // (4,.,.) "T"/"T", major/minor: 2/0
                {{4, 0, 1, 60, 0, 0},
                {4, 0, 1, 60, 0, 0}},

                // (5,.,.) "G"/nothing, major/minor: 3/0
                {{3, 0, 1, 60, 0, 0},
                {0, 0, 0, 0, 0, 0}},

                // (6,.,.), "A"/nothing, major/minor: 4/0
                {{1, 0, 1, 60, 0, 0},
                {0, 0, 0, 0, 0, 0}},

                // (7,.,.), "A"/nothing, major/minor: 5/0
                {{1, 0, 1, 60, 0, 0},
                {0, 0, 0, 0, 0, 0}},

                // (8,.,.), "C"/nothing, major/minor: 6/0
                {{2, 0, 1, 60, 0, 0},
                {0, 0, 0, 0, 0, 0}},

                // (9,.,.), "T"/nothing, major/minor: 7/0
                {{4, 0, 1, 60, 0, 0},
                {0, 0, 0, 0, 0, 0}},

                // (10,.,.), "-"/"G", major/minor: 8/0
                {{5, 0, 1, 60, 0, 0},
                {3, 0, 1, 60, 0, 0}},

                // (11,.,.), "A"/"A", major/minor: 9/0
                {{1, 0, 1, 60, 0, 0},
                {1, 0, 1, 60, 0, 0}},

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        },
        .positions_minor = {
            0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        .depth = torch::tensor(
            {2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
        },
        .read_ids_right = {
            "read_01",
            "read_03",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 10;
    const bool include_dwells{true};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{false};
    const HaplotagSource hap_source{HaplotagSource::UNPHASED};
    const std::unordered_map<std::string, int32_t> haplotags{};
    const double min_snp_accuracy{0.0};
    const bool clip_to_zero{true};
    const secondary::KadayashiOptions kadayashi_opt;

    // Test specific parameter.
    const bool row_per_read{false};

    // Run UUT.
    auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
            EncoderTestCase{"kadayashi feature encoder", false},
            EncoderTestCase{"legacy feature encoder", true},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
    EncoderReadAlignment encoder(
            temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
            read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
            clip_to_zero, right_align_insertions, include_haplotype_column, hap_source, phasing_bin,
            include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
    const Sample result = encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

    eval_sample(expected, result);
}

CATCH_TEST_CASE("synthetic_test_03-one_read_per_row", TEST_GROUP) {
    /**
     * Tests one read per row.
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        // Target (reference) sequences.
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGAACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "8M1D1M", "8^G1", "ACTGAACTA", "", {}, {}, 0},  // Full-span.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 0 /*flag*/,  60 /*mapq*/, "2M2I1M", "3", "ACAAT", "", {}, {}, 0},         // Reusable row, part 1.
            {"read_03", 0 /*tid*/, 8 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "2M", "2", "GA", "", {}, {}, 0},                // Reusable row, part 2.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    /**
     * Columns of the feature tensor are:
     *      base, qual, strand, mapq, dwell, haplotype [, dtype]
     *
     * NOTE: The base column is encoded in the following way:
     *      A = 1, C = 2, G = 3, T = 4, DEL = 5
     *
     *      The strand column is:
     *          FWD = 1, REV = 0
     */
    const Sample expected {
        .seq_id = 123,
        .features = torch::tensor(
            {
            // (0,.,.) "A"/"A", major/minor: 0/0
            {{1, 0, 1, 60, 0, 0},
            {1, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (1,.,.) "C"/"C", major/minor: 1/0
            {{2, 0, 1, 60, 0, 0},
            {2, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (2,.,.)  -> Insertion ("-"/"A"), major/minor: 1/1
            {{5, 0, 1, 60, 0, 0},
            {1, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (3,.,.)  -> Insertion ("-"/"A"), major/minor: 1/2
            {{5, 0, 1, 60, 0, 0},
            {1, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (4,.,.) "T"/"T", major/minor: 2/0
            {{4, 0, 1, 60, 0, 0},
            {4, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (5,.,.) "G"/nothing, major/minor: 3/0
            {{3, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (6,.,.), "A"/nothing, major/minor: 4/0
            {{1, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (7,.,.), "A"/nothing, major/minor: 5/0
            {{1, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (8,.,.), "C"/nothing, major/minor: 6/0
            {{2, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (9,.,.), "T"/nothing, major/minor: 7/0
            {{4, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}},

            // (10,.,.), "-"/"G", major/minor: 8/0
            {{5, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {3, 0, 1, 60, 0, 0}},

            // (11,.,.), "A"/"A", major/minor: 9/0
            {{1, 0, 1, 60, 0, 0},
            {0, 0, 0, 0, 0, 0},
            {1, 0, 1, 60, 0, 0}},

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        },
        .positions_minor = {
            0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        .depth = torch::tensor(
            {2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
            "__blank",
        },
        .read_ids_right = {
            "read_01",
            "__blank",
            "read_03",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 10;
    const bool include_dwells{true};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{false};
    const HaplotagSource hap_source{HaplotagSource::UNPHASED};
    const bool clip_to_zero{true};
    const std::unordered_map<std::string, int32_t> haplotags{};
    const double min_snp_accuracy{0.0};
    const secondary::KadayashiOptions kadayashi_opt;

    // Test specific parameter.
    const bool row_per_read{true};

    // Run UUT.
    auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
            EncoderTestCase{"kadayashi feature encoder", false},
            EncoderTestCase{"legacy feature encoder", true},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
    EncoderReadAlignment encoder(
            temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
            read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
            clip_to_zero, right_align_insertions, include_haplotype_column, hap_source, phasing_bin,
            include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
    const Sample result = encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

    eval_sample(expected, result);
}

CATCH_TEST_CASE("synthetic_test_04-max_reads", TEST_GROUP) {
    /**
     * Test picking at most one read from the input with `max_reads = 1`.
     * The input has 3 reads in total.
     *
     * IMPORTANT: The final tensor will still have insertion columns from read 3 even though
     *              max_reads == 1. This is because of how the matrix is filled, using the bam_mplp_auto.
     *              All elements of pileup column need to be processed, and only the reads which are within the
     *              max_reads rows will be kept.
     *
     *              TODO: Future: this means that for very deep coverages we might be getting a lot of empty columns
     *              even though we are keeping only max_reads == 100 by default. Perhaps it would
     *              be worth filtering out empty columns to reduce tensor sizes and improve speed in these cases.
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        // Target (reference) sequences.
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGAACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "8M1D1M", "8^G1", "ACTGAACTA", "", {}, {}, 0},  // Full-span.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 0 /*flag*/,  60 /*mapq*/, "2M2I1M", "3", "ACAAT", "", {}, {}, 0},         // Reusable row, part 1.
            {"read_03", 0 /*tid*/, 8 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "2M", "2", "GA", "", {}, {}, 0},                // Reusable row, part 2.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    /**
     * Columns of the feature tensor are:
     *      base, qual, strand, mapq, dwell, haplotype [, dtype]
     *
     * NOTE: The base column is encoded in the following way:
     *      A = 1, C = 2, G = 3, T = 4, DEL = 5
     *
     *      The strand column is:
     *          FWD = 1, REV = 0
     */
    const Sample expected {
        .seq_id = 123,
        .features = torch::tensor(
            {
            // (0,.,.) "A"/"A", major/minor: 0/0
            {{1, 0, 1, 60, 0, 0}},

            // (1,.,.) "C"/"C", major/minor: 1/0
            {{2, 0, 1, 60, 0, 0}},

            // (2,.,.) "C"/"C", major/minor: 1/1
            {{5, 0, 1, 60, 0, 0}},

            // (3,.,.) "C"/"C", major/minor: 1/2
            {{5, 0, 1, 60, 0, 0}},

            // (4,.,.) "T"/"T", major/minor: 2/0
            {{4, 0, 1, 60, 0, 0}},

            // (5,.,.) "G"/nothing, major/minor: 3/0
            {{3, 0, 1, 60, 0, 0}},

            // (6,.,.), "A"/nothing, major/minor: 4/0
            {{1, 0, 1, 60, 0, 0}},

            // (7,.,.), "A"/nothing, major/minor: 5/0
            {{1, 0, 1, 60, 0, 0}},

            // (8,.,.), "C"/nothing, major/minor: 6/0
            {{2, 0, 1, 60, 0, 0}},

            // (9,.,.), "T"/nothing, major/minor: 7/0
            {{4, 0, 1, 60, 0, 0}},

            // (10,.,.), "-"/"G", major/minor: 8/0
            {{5, 0, 1, 60, 0, 0}},

            // (11,.,.), "A"/"A", major/minor: 9/0
            {{1, 0, 1, 60, 0, 0}},

        }, torch::dtype(torch::kInt8)),
        .positions_major = {
            0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        },
        .positions_minor = {
            0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        .depth = torch::tensor(
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
        },
        .read_ids_right = {
            "read_01",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 10;
    const bool include_dwells{true};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{false};
    const HaplotagSource hap_source{HaplotagSource::UNPHASED};
    const bool clip_to_zero{true};
    const bool row_per_read{true};
    const std::unordered_map<std::string, int32_t> haplotags{};
    const double min_snp_accuracy{0.0};
    const secondary::KadayashiOptions kadayashi_opt;

    // Test a specific parameter.
    const int32_t max_reads{1};

    // Run UUT.
    auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
            EncoderTestCase{"kadayashi feature encoder", false},
            EncoderTestCase{"legacy feature encoder", true},
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
    EncoderReadAlignment encoder(
            temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
            read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
            clip_to_zero, right_align_insertions, include_haplotype_column, hap_source, phasing_bin,
            include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
    const Sample result = encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

    const std::vector<int64_t> shape(std::begin(result.features.sizes()),
                                     std::end(result.features.sizes()));

    eval_sample(expected, result);
}

CATCH_TEST_CASE("synthetic_test_05-haplotags", TEST_GROUP) {
    /**
     * Tests the haplotags assigned from the input unordered_map.
     * Auxiliary testing the same things as `synthetic_test_02` (insertion columns, deletion base values and multiple reads per row).
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        // Target (reference) sequences.
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGAACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "8M1D1M", "8^G1", "ACTGAACTA", "", {}, {}, 0},  // Full-span.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 0 /*flag*/,  60 /*mapq*/, "2M2I1M", "3", "ACAAT", "", {}, {}, 0},         // Reusable row, part 1.
            {"read_03", 0 /*tid*/, 8 /*pos*/, 0  /*flag*/, 60 /*mapq*/, "2M", "2", "GA", "", {}, {}, 0},                // Reusable row, part 2.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    /**
     * Columns of the feature tensor are:
     *      base, qual, strand, mapq, dwell, haplotype [, dtype]
     *
     * NOTE: The base column is encoded in the following way:
     *      A = 1, C = 2, G = 3, T = 4, DEL = 5
     *
     *      The strand column is:
     *          FWD = 1, REV = 0
     */
    const Sample expected {
        .seq_id = 123,
        .features = torch::tensor(
            {
                // (0,.,.) "A"/"A", major/minor: 0/0
                {{1, 0, 1, 60, 0, 1},
                {1, 0, 1, 60, 0, 123}},

                // (1,.,.) "C"/"C", major/minor: 1/0
                {{2, 0, 1, 60, 0, 1},
                {2, 0, 1, 60, 0, 123}},

                // (2,.,.)  -> Insertion ("-"/"A"), major/minor: 1/1
                {{5, 0, 1, 60, 0, 1},
                {1, 0, 1, 60, 0, 123}},

                // (3,.,.)  -> Insertion ("-"/"A"), major/minor: 1/2
                {{5, 0, 1, 60, 0, 1},
                {1, 0, 1, 60, 0, 123}},

                // (4,.,.) "T"/"T", major/minor: 2/0
                {{4, 0, 1, 60, 0, 1},
                {4, 0, 1, 60, 0, 123}},

                // (5,.,.) "G"/nothing, major/minor: 3/0
                {{3, 0, 1, 60, 0, 1},
                {0, 0, 0, 0, 0, 0}},

                // (6,.,.), "A"/nothing, major/minor: 4/0
                {{1, 0, 1, 60, 0, 1},
                {0, 0, 0, 0, 0, 0}},

                // (7,.,.), "A"/nothing, major/minor: 5/0
                {{1, 0, 1, 60, 0, 1},
                {0, 0, 0, 0, 0, 0}},

                // (8,.,.), "C"/nothing, major/minor: 6/0
                {{2, 0, 1, 60, 0, 1},
                {0, 0, 0, 0, 0, 0}},

                // (9,.,.), "T"/nothing, major/minor: 7/0
                {{4, 0, 1, 60, 0, 1},
                {0, 0, 0, 0, 0, 0}},

                // (10,.,.), "-"/"G", major/minor: 8/0
                {{5, 0, 1, 60, 0, 1},
                {3, 0, 1, 60, 0, 5}},

                // (11,.,.), "A"/"A", major/minor: 9/0
                {{1, 0, 1, 60, 0, 1},
                {1, 0, 1, 60, 0, 5}},

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        },
        .positions_minor = {
            0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        .depth = torch::tensor(
            {2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
        },
        .read_ids_right = {
            "read_01",
            "read_03",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 10;
    const bool include_dwells{true};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{false};
    const double min_snp_accuracy{0.0};
    const bool clip_to_zero{true};
    const bool row_per_read{false};
    const secondary::KadayashiOptions kadayashi_opt;

    CATCH_SECTION("Haplotags are precomputed and passed to the encoder") {
        // Test-specific parameters.
        const HaplotagSource hap_source{HaplotagSource::COMPUTE};
        const std::unordered_map<std::string, int32_t> haplotags{
                {"read_01", 1},
                {"read_02", 123},
                {"read_03", 5},
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }
}

CATCH_TEST_CASE("synthetic_test_06-calculate_read_alignment_fix_for_high_coverage_missing_reads",
                TEST_GROUP) {
    /**
     * Maximum number of pileup rows is set to 3x (max_reads = 3), but there are 4x of reads available.
     * Once read_03 exits, read_04 should be added after a 5-column gap (because we have hardcoded MIN_GAP = 5).
     *
     * Pileup matrix:
     *      Major       0   1   2   3   4   5   6   7   8   9   MAPQ
     *      Minor       0   0   0   0   0   0   0   0   0   0
     *      read_01     A   C   T   G   A   A   C   T   G   A   51
     *      read_02     A   C   T   G   A                       52
     *      read_03     A   C   T                               53
     *      read_04         C   T   G   A   A   C   T   G   A   54      <- Before the bugfix, this read would be ignored on major positions 8 and 9.
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGAACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 51 /*mapq*/, "10M" /*cigar*/, "10" /*MD*/, "ACTGAACTGA" /*seq*/, "" /*qual*/, {} /*dwell*/, {} /*hp*/, 0 /*NM*/},    // Full-span.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 16 /*flag*/, 52 /*mapq*/, "5M", "0A4", "ACTGA", "", {}, 3, 1},                                    // Left-flank only. Reverse. NM = 1 and cigar = 5M make snp_qv = 6.98.
            {"read_03", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 53 /*mapq*/, "3M", "0A0C1", "ACT", "", {5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 5, 2},  // This should share the row with the next one. NM = 2 and cigar = 3M make snp_qv = 1.76.
            {"read_04", 0 /*tid*/, 1 /*pos*/, 0  /*flag*/, 54 /*mapq*/, "9M", "0C0T0G6", "CTGAACTGA", "123456789", {}, {}, 3},                  // Contained. Has quals. NM = 3 and cigar = 9M make snp_qv = 4.77.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    const Sample expected_total {
        .seq_id = 123,
        // Features tensor shape: [pos, coverage, features] -> [10, 5, 7]
        // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
        .features = torch::tensor(
            {
                // (0,.,.)
                {{1, 0, 1, 51, 0, 0, 60},       // read_01
                 {1, 0, 0, 52, 0, 3, 7},        // read_02
                 {1, 0, 1, 53, 4, 5, 2},        // read_03
                },

                // (1,.,.)
                {{2, 0,  1, 51, 0, 0, 60},
                 {2, 0,  0, 52, 0, 3,  7},
                 {2, 0,  1, 53, 5, 5,  2},
                },

                // (2,.,.)
                {{4, 0,  1, 51, 0, 0, 60},
                 {4, 0,  0, 52, 0, 3, 7},
                 {4, 0,  1, 53, 2, 5, 2},
                },

                // (3,.,.)
                {{3, 0,  1, 51, 0, 0, 60},
                 {3, 0,  0, 52, 0, 3,  7},
                 {0, 0,  0,  0, 0, 0,  0}       // read_03 exits
                },

                // (4,.,.)
                {{1,  0,  1, 51, 0, 0, 60},
                 {1,  0,  0, 52, 0, 3,  7},
                 {0,  0,  0,  0, 0, 0,  0},
                },

                // (5,.,.)
                {{1, 0,  1, 51, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},      // read_02 exits
                 {0, 0,  0,  0, 0, 0,  0},
                },

                // (6,.,.)
                {{2, 0,  1, 51, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                },

                // (7,.,.)
                {{4, 0,  1,51, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                },

                // (8,.,.)
                {{3, 0,  1, 51, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 56,  1, 54, 0, 0,  5},      // Remainder of read_04 enters on the last row because read_03 exited MIN_GAP=5 columns ago.
                },

                // (9,.,.)
                {{1, 0, 1, 51, 0, 0, 60},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 57, 1, 54, 0, 0, 5},
                },

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        },
        .positions_minor = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        .depth = torch::tensor(
            {3, 3, 3, 2, 2, 1, 1, 1, 2, 2},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
            "read_03",
        },
        .read_ids_right = {
            "read_01",
            "__blank",
            "read_04",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const bool row_per_read{false};
    const double min_snp_accuracy{0.0};
    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 10;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;
    const bool include_dwells{true};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{true};
    const HaplotagSource hap_source{HaplotagSource::BAM_HAP_TAG};
    const bool clip_to_zero{true};

    CATCH_SECTION("Limit coverage to 3x. read_04 should enter 5 columns after read_03 exits.") {
        // Test specific parameters.
        const int32_t max_reads{3};

        // Expected results for this test, including the dwell and the haplotag columns. add a column of zeros for the haplotype for each read.
        const Sample& expected = expected_total;

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);

        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }
}

CATCH_TEST_CASE("synthetic_test_07-populate_draft_seq_tensor", TEST_GROUP) {
    // clang-format off
    const std::vector<std::string> reference_sequences{
        "ACTGAACTGA",
        "ACTgaacTGA",
        "AGGATTCNNNGABACG",
    };

    // clang-format off
    const std::vector<
        std::tuple<int64_t, std::vector<int64_t>, std::vector<int64_t> >
        //Each record is (sequence_id, positions_major, positions_minor)
    > ranges{
        {
            // All of sequence 0
            0,
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
            { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
        },
        {
            // Subsequence of sequence 0 with gaps
            0,
            { 2, 3, 3, 3, 4, 5, 6, 7, 8 },
            { 0, 0, 1, 2, 0, 0, 0, 0, 0 }
        },
        {
            // Subsequence of sequence 1 with gaps and lower case characters
            1,
            { 2, 3, 3, 3, 4, 5, 6, 7, 8 },
            { 0, 0, 1, 2, 0, 0, 0, 0, 0 }
        },
        {
            // Subsequence of sequence 2 with gaps and ambiguity characters
            2,
            { 6, 6, 7, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15 },
            { 0, 1, 0, 0, 1, 2, 3, 0,  0,  0,  0,  0,  0,  0 }
        },
        {
            // Subsequence of sequence 2 starting on a minor position
            2,
            { 1, 1, 2, 3, 4 },
            { 1, 2, 0, 0, 0 }
        },
        {
            // Entire sample is in an insertion
            2,
            { 1, 1, 1, 1, 1, 1 },
            { 2, 3, 4, 5, 6, 7 }
        }
    };

    // Expected results.
    const std::vector<torch::Tensor> expected_results{
        torch::tensor(
            // A  C  T  G  A  A  C  T  G  A
            {  1, 2, 4, 3, 1, 1, 2, 4, 3, 1}, torch::kInt8),
        torch::tensor(
            // T  G  -  -  A  A  C  T  G
            {  4, 3, 5, 5, 1, 1, 2, 4, 3}, torch::kInt8),
        torch::tensor(
            // T  G  -  -  A  A  C  T  G
            {  4, 3, 5, 5, 1, 1, 2, 4, 3}, torch::kInt8),
        torch::tensor(
            // C  -  N  N  -  -  -  N  G  A  B  A  C  G
            {  2, 5, 5, 5, 5, 5, 5, 5, 3, 1, 5, 1, 2, 3}, torch::kInt8),
        torch::tensor(
            // -  -  G  A  T
            {  5, 5, 3, 1, 4}, torch::kInt8),
        torch::tensor(
            // -  -  -  -  -  -
            {  5, 5, 5, 5, 5, 5}, torch::kInt8)
    };
    // clang-format on

    for (int64_t i = 0; i < std::ssize(ranges); ++i) {
        CATCH_SECTION("Example " + std::to_string(i)) {
            auto [seq_id, positions_major, positions_minor] = ranges[i];
            torch::Tensor result = draft_encoding_from_seq(positions_major, positions_minor,
                                                           reference_sequences[seq_id]);
            CATCH_CHECK(result.equal(expected_results[i]));
        }
    }
}

CATCH_TEST_CASE("synthetic_test_08-zero_coverage_trimming_start", TEST_GROUP) {
    /**
     * This test mirrors synthetic_test_01 except that the reads do not span the entire reference window.
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        // two added bases at the start of the reference
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "AAACTGAACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/,  2 /*pos*/, 0  /*flag*/, 50 /*mapq*/, "10M" /*cigar*/, "10" /*MD*/, "ACTGAACTGA" /*seq*/, "" /*qual*/, {} /*dwell*/, {} /*hp*/, 0 /*NM*/},    // Full-span.
            {"read_01", 0 /*tid*/,  2 /*pos*/, 0  /*flag*/, 51 /*mapq*/, "10M", "10", "ACTGAACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_01", 0 /*tid*/,  2 /*pos*/, 0  /*flag*/, 52 /*mapq*/, "10M", "10", "ACTGAACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_02", 0 /*tid*/,  2 /*pos*/, 16 /*flag*/, 53 /*mapq*/, "5M", "0A4", "ACTGA", "", {}, 3, 1},                                    // Left-flank only. Reverse. NM = 1 and cigar = 5M make snp_qv = 6.98.
            {"read_03", 0 /*tid*/,  2 /*pos*/, 0  /*flag*/, 54 /*mapq*/, "3M", "0A0C1", "ACT", "", {5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 5, 2},  // This should share the row with the next one. NM = 2 and cigar = 3M make snp_qv = 1.76.
            {"read_04", 0 /*tid*/, 10 /*pos*/, 0  /*flag*/, 55 /*mapq*/, "2M", "0G1", "GA", "", {}, {}, 1},                                      // This should share the row with the previous.
            {"read_05", 0 /*tid*/,  3 /*pos*/, 0  /*flag*/, 56 /*mapq*/, "8M", "0C0T0G5", "CTGAACTG", "1234567890", {}, {}, 3},                  // Contained. Has quals. NM = 3 and cigar = 8M make snp_qv = 4.26.
            {"read_06", 0 /*tid*/,  7 /*pos*/, 0  /*flag*/, 57 /*mapq*/, "5M", "0A0C0T0G0A0", "ACTGA", "", {}, {}, 5},                           // Right-flank only.
            {"read_06", 0 /*tid*/,  7 /*pos*/, 0  /*flag*/, 58 /*mapq*/, "5M", "5", "ACTGA", "", {}, {}, 0},                                     // Duplicate.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    const Sample expected_total {
        .seq_id = 123,
        // Features tensor shape: [pos, coverage, features] -> [10, 5, 7]
        // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
        .features = torch::tensor(
            {
                // (2,.,.)
                {{1, 0, 1, 50, 0, 0, 60},
                 {1, 0, 0, 53, 0, 3, 7},
                 {1, 0, 1, 54, 4, 5, 2},
                 {0, 0, 0,  0, 0, 0, 0},
                 {0, 0, 0,  0, 0, 0, 0}},

                // (3,.,.)
                {{2, 0,  1, 50, 0, 0, 60},
                 {2, 0,  0, 53, 0, 3,  7},
                 {2, 0,  1, 54, 5, 5,  2},
                 {2, 49, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (4,.,.)
                {{4, 0,  1, 50, 0, 0, 60},
                 {4, 0,  0, 53, 0, 3, 7},
                 {4, 0,  1, 54, 2, 5, 2},
                 {4, 50, 1, 56, 0, 0, 4},
                 {0, 0,  0,  0, 0, 0, 0}},

                // (5,.,.)
                {{3, 0,  1, 50, 0, 0, 60},
                 {3, 0,  0, 53, 0, 3,  7},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 51, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (6,.,.)
                {{1,  0,  1, 50, 0, 0, 60},
                 {1,  0,  0, 53, 0, 3,  7},
                 {0,  0,  0,  0, 0, 0,  0},
                 {1, 52,  1, 56, 0, 0,  4},
                 {0,  0,  0,  0, 0, 0,  0}},

                // (7,.,.)
                {{1, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {1, 53, 1, 56, 0, 0,  4},
                 {1, 0,  1, 57, 0, 0,  0}},

                // (8,.,.)
                {{2, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {2, 54, 1, 56, 0, 0,  4},
                 {2, 0,  1, 57, 0, 0,  0}},

                // (9,.,.)
                {{4, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {4, 55, 1, 56, 0, 0,  4},
                 {4, 0,  1, 57, 0, 0,  0}},

                // (10,.,.)
                {{3, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 0,  1, 55, 0, 0,  3},
                 {3, 56, 1, 56, 0, 0,  4},
                 {3, 0,  1, 57, 0, 0,  0}},

                // (11,.,.)
                {{1, 0, 1, 50, 0, 0, 60},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 0, 1, 55, 0, 0,  3},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 0, 1, 57, 0, 0,  0}},

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        },
        .positions_minor = {
            0, 0, 0, 0, 0, 0, 0, 0,  0,  0
        },
        .depth = torch::tensor(
            {3, 4, 4, 3, 3, 3, 3, 3, 4, 3},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
            "read_03",
            "__blank",
            "__blank",
        },
        .read_ids_right = {
            "read_01",
            "__blank",
            "read_04",
            "__blank",
            "read_06",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const double min_snp_accuracy{0.0};

    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 12;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;

    CATCH_SECTION(
            "No dwell column, no hap column, no snp_qv column. Only the base features (base, qual, "
            "strand, mapq)."
            "Tests the entire output Sample: feature tensor, depth tensor, positions major/minor, "
            "and read IDs. Include dwell col, no hap col") {
        // Test specific parameters.
        const bool include_dwells{false};
        const bool include_haplotype_column{false};
        const bool include_snp_qv_column{false};
        const HaplotagSource hap_source{HaplotagSource::UNPHASED};
        const bool clip_to_zero{true};

        // Expected results for this test: drop the dwell column from the last dimension.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index({"...", torch::indexing::Slice(0, 4)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }
}

CATCH_TEST_CASE("synthetic_test_09-zero_coverage_trimming_end", TEST_GROUP) {
    /**
     * This test mirrors synthetic_test_01 except that the reads do not span the entire reference window.
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        // two added bases at the end of the reference
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGAACTGACG"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 50 /*mapq*/, "10M" /*cigar*/, "10" /*MD*/, "ACTGAACTGA" /*seq*/, "" /*qual*/, {} /*dwell*/, {} /*hp*/, 0 /*NM*/},    // Full-span.
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 51 /*mapq*/, "10M", "10", "ACTGAACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 52 /*mapq*/, "10M", "10", "ACTGAACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 16 /*flag*/, 53 /*mapq*/, "5M", "0A4", "ACTGA", "", {}, 3, 1},                                    // Left-flank only. Reverse. NM = 1 and cigar = 5M make snp_qv = 6.98.
            {"read_03", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 54 /*mapq*/, "3M", "0A0C1", "ACT", "", {5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 5, 2},  // This should share the row with the next one. NM = 2 and cigar = 3M make snp_qv = 1.76.
            {"read_04", 0 /*tid*/, 8 /*pos*/, 0  /*flag*/, 55 /*mapq*/, "2M", "0G1", "GA", "", {}, {}, 1},                                      // This should share the row with the previous.
            {"read_05", 0 /*tid*/, 1 /*pos*/, 0  /*flag*/, 56 /*mapq*/, "8M", "0C0T0G5", "CTGAACTG", "1234567890", {}, {}, 3},                  // Contained. Has quals. NM = 3 and cigar = 8M make snp_qv = 4.26.
            {"read_06", 0 /*tid*/, 5 /*pos*/, 0  /*flag*/, 57 /*mapq*/, "5M", "0A0C0T0G0A0", "ACTGA", "", {}, {}, 5},                           // Right-flank only.
            {"read_06", 0 /*tid*/, 5 /*pos*/, 0  /*flag*/, 58 /*mapq*/, "5M", "5", "ACTGA", "", {}, {}, 0},                                     // Duplicate.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    const Sample expected_total {
        .seq_id = 123,
        // Features tensor shape: [pos, coverage, features] -> [10, 5, 7]
        // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
        .features = torch::tensor(
            {
                // (0,.,.)
                {{1, 0, 1, 50, 0, 0, 60},
                 {1, 0, 0, 53, 0, 3, 7},
                 {1, 0, 1, 54, 4, 5, 2},
                 {0, 0, 0,  0, 0, 0, 0},
                 {0, 0, 0,  0, 0, 0, 0}},

                // (1,.,.)
                {{2, 0,  1, 50, 0, 0, 60},
                 {2, 0,  0, 53, 0, 3,  7},
                 {2, 0,  1, 54, 5, 5,  2},
                 {2, 49, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (2,.,.)
                {{4, 0,  1, 50, 0, 0, 60},
                 {4, 0,  0, 53, 0, 3, 7},
                 {4, 0,  1, 54, 2, 5, 2},
                 {4, 50, 1, 56, 0, 0, 4},
                 {0, 0,  0,  0, 0, 0, 0}},

                // (3,.,.)
                {{3, 0,  1, 50, 0, 0, 60},
                 {3, 0,  0, 53, 0, 3,  7},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 51, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (4,.,.)
                {{1,  0,  1, 50, 0, 0, 60},
                 {1,  0,  0, 53, 0, 3,  7},
                 {0,  0,  0,  0, 0, 0,  0},
                 {1, 52,  1, 56, 0, 0,  4},
                 {0,  0,  0,  0, 0, 0,  0}},

                // (5,.,.)
                {{1, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {1, 53, 1, 56, 0, 0,  4},
                 {1, 0,  1, 57, 0, 0,  0}},

                // (6,.,.)
                {{2, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {2, 54, 1, 56, 0, 0,  4},
                 {2, 0,  1, 57, 0, 0,  0}},

                // (7,.,.)
                {{4, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {4, 55, 1, 56, 0, 0,  4},
                 {4, 0,  1, 57, 0, 0,  0}},

                // (8,.,.)
                {{3, 0,  1, 50, 0, 0, 60},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 0,  1, 55, 0, 0,  3},
                 {3, 56, 1, 56, 0, 0,  4},
                 {3, 0,  1, 57, 0, 0,  0}},

                // (9,.,.)
                {{1, 0, 1, 50, 0, 0, 60},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 0, 1, 55, 0, 0,  3},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 0, 1, 57, 0, 0,  0}},

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        },
        .positions_minor = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        },
        .depth = torch::tensor(
            {3, 4, 4, 3, 3, 3, 3, 3, 4, 3},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
            "read_03",
            "__blank",
            "__blank",
        },
        .read_ids_right = {
            "read_01",
            "__blank",
            "read_04",
            "__blank",
            "read_06",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const double min_snp_accuracy{0.0};

    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 12;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;

    CATCH_SECTION(
            "No dwell column, no hap column, no snp_qv column. Only the base features (base, qual, "
            "strand, mapq)."
            "Tests the entire output Sample: feature tensor, depth tensor, positions major/minor, "
            "and read IDs. Include dwell col, no hap col") {
        // Test specific parameters.
        const bool include_dwells{false};
        const bool include_haplotype_column{false};
        const bool include_snp_qv_column{false};
        const HaplotagSource hap_source{HaplotagSource::UNPHASED};
        const bool clip_to_zero{true};

        // Expected results for this test: drop the dwell column from the last dimension.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index({"...", torch::indexing::Slice(0, 4)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }
}

CATCH_TEST_CASE("synthetic_test_10-zero_coverage_internal", TEST_GROUP) {
    /**
     * This test mirrors synthetic_test_01 except that the reads do not span the entire reference window.
     */

    const auto temp_dir = make_temp_dir("encoder_read_aln_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    // Create the input test BAM/ref FASTA.
    {
        // clang-format off
        // two added bases in the middle of the reference
        const std::vector<std::pair<std::string, std::string>> targets{
                {"contig_1", "ACTGATTACTGA"},
        };
        const std::vector<BamRecord> records{
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 50 /*mapq*/, "5M" /*cigar*/, "5" /*MD*/, "ACTGA" /*seq*/, "" /*qual*/, {} /*dwell*/, {} /*hp*/, 0 /*NM*/},    // Full-span.
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 51 /*mapq*/, "5M", "5", "ACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_01", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 52 /*mapq*/, "5M", "5", "ACTGA", "", {}, {}, 0},                              // Duplicate.
            {"read_02", 0 /*tid*/, 0 /*pos*/, 16 /*flag*/, 53 /*mapq*/, "5M", "0A4", "ACTGA", "", {}, 3, 1},                                    // Left-flank only. Reverse. NM = 1 and cigar = 5M make snp_qv = 6.98.
            {"read_03", 0 /*tid*/, 0 /*pos*/, 0  /*flag*/, 54 /*mapq*/, "3M", "0A0C1", "ACT", "", {5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 5, 2},  // This should share the row with the next one. NM = 2 and cigar = 3M make snp_qv = 1.76.
            {"read_04", 0 /*tid*/, 10 /*pos*/, 0  /*flag*/, 55 /*mapq*/, "2M", "0G1", "GA", "", {}, {}, 1},                                      // This should share the row with the previous.
            {"read_05", 0 /*tid*/, 1 /*pos*/, 0  /*flag*/, 56 /*mapq*/, "4M", "0C0T0G1", "CTGA", "1234", {}, {}, 3},                  // Contained. Has quals. NM = 3 and cigar = 8M make snp_qv = 4.26.
            {"read_06", 0 /*tid*/, 7 /*pos*/, 0  /*flag*/, 57 /*mapq*/, "5M", "0A0C0T0G0A0", "ACTGA", "", {}, {}, 5},                           // Right-flank only.
            {"read_06", 0 /*tid*/, 7 /*pos*/, 0  /*flag*/, 58 /*mapq*/, "5M", "5", "ACTGA", "", {}, {}, 0},                                     // Duplicate.
        };
        // clang-format on

        write_bam(temp_in_bam_fn, targets, records);
        write_ref(temp_in_ref_fn, targets);
    }

    // Expected results.
    // clang-format off
    const Sample expected_total {
        .seq_id = 123,
        // Features tensor shape: [pos, coverage, features] -> [10, 5, 7]
        // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
        .features = torch::tensor(
            {
                // (0,.,.)
                {{1, 0, 1, 50, 0, 0, 60},
                 {1, 0, 0, 53, 0, 3, 7},
                 {1, 0, 1, 54, 4, 5, 2},
                 {0, 0, 0,  0, 0, 0, 0},
                 {0, 0, 0,  0, 0, 0, 0}},

                // (1,.,.)
                {{2, 0,  1, 50, 0, 0, 60},
                 {2, 0,  0, 53, 0, 3,  7},
                 {2, 0,  1, 54, 5, 5,  2},
                 {2, 49, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (2,.,.)
                {{4, 0,  1, 50, 0, 0, 60},
                 {4, 0,  0, 53, 0, 3, 7},
                 {4, 0,  1, 54, 2, 5, 2},
                 {4, 50, 1, 56, 0, 0, 4},
                 {0, 0,  0,  0, 0, 0, 0}},

                // (3,.,.)
                {{3, 0,  1, 50, 0, 0, 60},
                 {3, 0,  0, 53, 0, 3,  7},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 51, 1, 56, 0, 0,  4},
                 {0, 0,  0,  0, 0, 0,  0}},

                // (4,.,.)
                {{1,  0,  1, 50, 0, 0, 60},
                 {1,  0,  0, 53, 0, 3,  7},
                 {0,  0,  0,  0, 0, 0,  0},
                 {1, 52,  1, 56, 0, 0,  4},
                 {0,  0,  0,  0, 0, 0,  0}},

                // (7,.,.)
                {{0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {1, 0,  1, 57, 0, 0,  0}},

                // (8,.,.)
                {{0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {2, 0,  1, 57, 0, 0,  0}},

                // (9,.,.)
                {{0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {4, 0,  1, 57, 0, 0,  0}},

                // (10,.,.)
                {{3, 0,  1, 55, 0, 0,  3},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {0, 0,  0,  0, 0, 0,  0},
                 {3, 0,  1, 57, 0, 0,  0}},

                // (11,.,.)
                {{1, 0, 1, 55, 0, 0,  3},
                 {0, 0, 0,  0, 0, 0,  0},
                 {0, 0, 0,  0, 0, 0,  0},
                 {0, 0, 0,  0, 0, 0,  0},
                 {1, 0, 1, 57, 0, 0,  0}},

            }, torch::dtype(torch::kInt8)
        ),
        .positions_major = {
            0, 1, 2, 3, 4, 7, 8, 9, 10, 11
        },
        .positions_minor = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        },
        .depth = torch::tensor(
            {3, 4, 4, 3, 3, 1, 1, 1, 2, 2},
            torch::dtype(torch::kInt32)
        ),
        .read_ids_left = {
            "read_01",
            "read_02",
            "read_03",
            "__blank",
            "__blank",
        },
        .read_ids_right = {
            "read_04",
            "__blank",
            "__blank",
            "__blank",
            "read_06",
        },
    };
    // clang-format on

    // Parameters, fixed for this test.
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const double min_snp_accuracy{0.0};

    const bool right_align_insertions{false};
    const std::optional<std::filesystem::path> phasing_bin{};
    const std::string ref_name{"contig_1"};
    const int32_t ref_id = 123;
    const int64_t ref_start = 0;
    const int64_t ref_end = 12;
    const std::unordered_map<std::string, int32_t> haplotags{};
    const secondary::KadayashiOptions kadayashi_opt;

    CATCH_SECTION(
            "No dwell column, no hap column, no snp_qv column. Only the base features (base, qual, "
            "strand, mapq)."
            "Tests the entire output Sample: feature tensor, depth tensor, positions major/minor, "
            "and read IDs. Include dwell col, no hap col") {
        // Test specific parameters.
        const bool include_dwells{false};
        const bool include_haplotype_column{false};
        const bool include_snp_qv_column{false};
        const HaplotagSource hap_source{HaplotagSource::UNPHASED};
        const bool clip_to_zero{true};

        // Expected results for this test: drop the dwell column from the last dimension.
        const Sample expected{
                .seq_id = expected_total.seq_id,
                .features = expected_total.features.index({"...", torch::indexing::Slice(0, 4)}),
                .positions_major = expected_total.positions_major,
                .positions_minor = expected_total.positions_minor,
                .depth = expected_total.depth,
                .read_ids_left = expected_total.read_ids_left,
                .read_ids_right = expected_total.read_ids_right,
        };

        // Run UUT.
        auto [test_case] = GENERATE_REF(table<EncoderTestCase>({
                EncoderTestCase{"kadayashi feature encoder", false},
                EncoderTestCase{"legacy feature encoder", true},
        }));

        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);
        EncoderReadAlignment encoder(
                temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                phasing_bin, include_snp_qv_column, kadayashi_opt, test_case.legacy_feat_encoder);
        const Sample result =
                encoder.encode_region(ref_name, ref_start, ref_end, ref_id, haplotags);

        eval_sample(expected, result);
    }
}

CATCH_TEST_CASE("EncoderReadAlignment::collate preserves padded batch shape after clamping",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_collate_test");
    const auto temp_in_ref_fn = temp_dir.m_path / "in.ref.fasta";
    const auto temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    const std::vector<std::pair<std::string, std::string>> targets{{"contig_1", "ACTGAACTGA"}};
    const std::vector<BamRecord> records{
            {"read_01", 0, 0, 0, 50, "10M", "10", "ACTGAACTGA", "", {}, {}, 0},
    };
    write_bam(temp_in_bam_fn, targets, records);
    write_ref(temp_in_ref_fn, targets);

    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const double min_snp_accuracy{0.0};
    const bool row_per_read{false};
    const bool include_dwells{false};
    const bool clip_to_zero{true};
    const bool right_align_insertions{false};
    const bool include_haplotype_column{false};
    const HaplotagSource hap_source{HaplotagSource::BAM_HAP_TAG};
    const std::optional<std::filesystem::path> phasing_bin{};
    const bool include_snp_qv_column{false};
    const KadayashiOptions kadayashi_opt{};

    EncoderReadAlignment encoder(temp_in_ref_fn, temp_in_bam_fn, dtypes, tag_name, tag_value,
                                 tag_keep_missing, read_group, min_mapq, max_reads,
                                 min_snp_accuracy, row_per_read, include_dwells, clip_to_zero,
                                 right_align_insertions, include_haplotype_column, hap_source,
                                 phasing_bin, include_snp_qv_column, kadayashi_opt, false);

    const at::TensorOptions opts = at::TensorOptions().dtype(at::kChar).device(at::kCPU);
    std::vector<at::Tensor> batch;
    batch.emplace_back(at::full({300, 1, 1}, -1, opts));
    batch.emplace_back(at::full({300, 2, 1}, 7, opts));

    const at::Tensor result = encoder.collate(std::move(batch), false);

    CATCH_CHECK(result.sizes().vec() == std::vector<int64_t>{2, 300, 2, 1});
    CATCH_CHECK(result.index({0}).max().item<int64_t>() == 0);
    CATCH_CHECK(result.index({1, torch::indexing::Slice(), torch::indexing::Slice(0, 2),
                              torch::indexing::Slice()})
                        .min()
                        .item<int64_t>() == 7);
}

namespace {

// Creates an encoder over a trivial BAM/reference. Only used to access merge_adjacent_samples().
EncoderReadAlignment make_merge_test_encoder(const std::filesystem::path& dir) {
    const auto ref_fn = dir / "in.ref.fasta";
    const auto bam_fn = dir / "in.aln.bam";
    const std::vector<std::pair<std::string, std::string>> targets{{"contig_1", "ACTGAACTGA"}};
    const std::vector<BamRecord> records{
            {"read_01", 0, 0, 0, 50, "10M", "10", "ACTGAACTGA", "", {}, {}, 0}};
    write_bam(bam_fn, targets, records);
    write_ref(ref_fn, targets);
    return EncoderReadAlignment(ref_fn, bam_fn, {}, "", 0, false, "", 1, 100, 0.0, false, false,
                                true, false, false, HaplotagSource::BAM_HAP_TAG, {}, false,
                                KadayashiOptions{}, false);
}

// Creates a read-level sample where features[p, r, 0] == row_values[r] for every position p.
Sample make_read_level_sample(const int64_t start,
                              const int64_t num_pos,
                              const std::vector<int8_t>& row_values,
                              std::vector<std::string> read_ids_left,
                              std::vector<std::string> read_ids_right) {
    const int64_t depth = std::ssize(row_values);
    Sample sample;
    sample.seq_id = 0;
    sample.features =
            at::tensor(row_values, at::kChar).view({1, depth, 1}).expand({num_pos, -1, 1}).clone();
    sample.positions_major.resize(num_pos);
    std::iota(std::begin(sample.positions_major), std::end(sample.positions_major), start);
    sample.positions_minor.assign(num_pos, 0);
    sample.depth = at::ones({num_pos}, at::kLong);
    sample.read_ids_left = std::move(read_ids_left);
    sample.read_ids_right = std::move(read_ids_right);
    sample.validate();
    return sample;
}

// Expected features for a merged sample: each (num_pos, row_values) block is stacked along dim 0
// and zero-padded along dim 1 up to the maximum depth.
at::Tensor expected_features(const std::vector<std::pair<int64_t, std::vector<int8_t>>>& blocks) {
    std::vector<at::Tensor> parts;
    parts.reserve(std::size(blocks));
    for (const auto& [num_pos, row_values] : blocks) {
        parts.emplace_back(at::tensor(row_values, at::kChar)
                                   .view({1, std::ssize(row_values), 1})
                                   .expand({num_pos, -1, 1})
                                   .clone());
    }
    return at::cat(parts, 0);
}

using Ids = std::vector<std::string>;

}  // namespace

CATCH_TEST_CASE("EncoderReadAlignment::merge_adjacent_samples: single sample is unchanged",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    const Sample in = make_read_level_sample(0, 3, {1, 2}, {"a", "b"}, {"x", "y"});
    const auto result = encoder.merge_adjacent_samples({in});
    const at::Tensor expected = expected_features({{3, {1, 2}}});

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(result[0].read_ids_left == in.read_ids_left);
    CATCH_CHECK(result[0].read_ids_right == in.read_ids_right);
    CATCH_CHECK(at::equal(result[0].features, in.features));
    CATCH_CHECK(at::equal(result[0].features, expected));
}

CATCH_TEST_CASE("EncoderReadAlignment::merge_adjacent_samples: same read order", TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // Right IDs of each chunk match the left IDs of the next one: no reordering needed.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {1, 2, 3}, {"a", "b", "c"}, {"d", "e", "f"}),
            make_read_level_sample(2, 3, {4, 5, 6}, {"d", "e", "f"}, {"g", "h", "i"}),
            make_read_level_sample(5, 1, {7, 8, 9}, {"g", "h", "i"}, {"j", "k", "l"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{0, 1, 2, 3, 4, 5});
    CATCH_CHECK(result[0].read_ids_left == Ids{"a", "b", "c"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"j", "k", "l"});
    CATCH_CHECK(at::equal(result[0].features,
                          expected_features({{2, {1, 2, 3}}, {3, {4, 5, 6}}, {1, {7, 8, 9}}})));
}

CATCH_TEST_CASE("EncoderReadAlignment::merge_adjacent_samples: rows are reordered at same depth",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // The second chunk lists the same reads (a, b, c) in a different row order (c, a, b).
    // Row values equal the read index of the first chunk (10, 11, 12) so the reorder is visible.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {10, 11, 12}, {"l0", "l1", "l2"}, {"a", "b", "c"}),
            make_read_level_sample(2, 2, {12, 10, 11}, {"c", "a", "b"}, {"x", "y", "z"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(result[0].read_ids_left == Ids{"l0", "l1", "l2"});
    // Out IDs of the second chunk are permuted along with its rows.
    CATCH_CHECK(result[0].read_ids_right == Ids{"y", "z", "x"});
    CATCH_CHECK(at::equal(result[0].features,
                          expected_features({{2, {10, 11, 12}}, {2, {10, 11, 12}}})));
}

CATCH_TEST_CASE("EncoderReadAlignment::merge_adjacent_samples: depth shrinks in a later chunk",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // Read c ends at the boundary, so the second chunk has one row fewer.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {10, 11, 12}, {"l0", "l1", "l2"}, {"a", "b", "c"}),
            make_read_level_sample(2, 2, {10, 11}, {"a", "b"}, {"x", "y"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK(result[0].features.size(1) == 3);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(result[0].read_ids_left == Ids{"l0", "l1", "l2"});
    CATCH_CHECK(std::ssize(result[0].read_ids_right) == 3);
    CATCH_CHECK(result[0].read_ids_right[0] == "x");
    CATCH_CHECK(result[0].read_ids_right[1] == "y");
    CATCH_CHECK(at::equal(result[0].features,
                          expected_features({{2, {10, 11, 12}}, {2, {10, 11, 0}}})));
}

CATCH_TEST_CASE("EncoderReadAlignment::merge_adjacent_samples: depth grows in a later chunk",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // A new read (c) starts in the second chunk. The merged tensor is padded to depth 3, and
    // read_ids_left must be consistent with that depth.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {10, 11}, {"l0", "l1"}, {"a", "b"}),
            make_read_level_sample(2, 2, {10, 11, 12}, {"a", "b", "c"}, {"x", "y", "z"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK(result[0].features.size(1) == 3);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(result[0].read_ids_left == Ids{"l0", "l1", "__blank"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"x", "y", "z"});
    CATCH_CHECK(at::equal(result[0].features,
                          expected_features({{2, {10, 11, 0}}, {2, {10, 11, 12}}})));
}

CATCH_TEST_CASE("EncoderReadAlignment::merge_adjacent_samples: depth grows in a middle chunk",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // Max depth is reached in the middle chunk; both boundary ID vectors must follow it.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 1, {10, 11}, {"l0", "l1"}, {"a", "b"}),
            make_read_level_sample(1, 1, {10, 11, 12}, {"a", "b", "c"}, {"p", "q", "r"}),
            make_read_level_sample(2, 1, {10, 11}, {"p", "q"}, {"x", "y"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK(result[0].features.size(1) == 3);
    CATCH_CHECK(
            at::equal(result[0].features,
                      expected_features({{1, {10, 11, 0}}, {1, {10, 11, 12}}, {1, {10, 11, 0}}})));
    CATCH_CHECK(result[0].read_ids_left == Ids{"l0", "l1", "__blank"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"x", "y", "__blank"});
}

CATCH_TEST_CASE("EncoderReadAlignment::merge_adjacent_samples: non-adjacent samples stay separate",
                TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    const std::vector<Sample> in{
            make_read_level_sample(0, 2, {10, 11}, {"a", "b"}, {"c", "d"}),
            make_read_level_sample(5, 2, {12, 13, 14}, {"e", "f", "g"}, {"h", "i", "j"}),
    };

    // Gap between positions 2 and 5, so the chunks must not be merged.
    const auto result = encoder.merge_adjacent_samples(in);

    CATCH_REQUIRE(std::size(result) == 2);
    CATCH_CHECK(at::equal(result[0].features, in[0].features));
    CATCH_CHECK(result[0].read_ids_right == Ids{"c", "d"});
    CATCH_CHECK(at::equal(result[1].features, in[1].features));
    CATCH_CHECK(result[1].read_ids_right == Ids{"h", "i", "j"});
}

CATCH_TEST_CASE(
        "EncoderReadAlignment::merge_adjacent_samples: empty left ID in 2nd chunk matches an "
        "unmatched row of the 1st chunk",
        TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // Read B has no named counterpart in the second chunk, but the second chunk has one empty
    // left ID, which takes the B row. No extra depth is needed.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {1, 2, 3}, {"iA", "iB", "iC"}, {"A", "B", "C"}),
            make_read_level_sample(2, 2, {13, 12, 11}, {"C", "__blank", "A"},
                                   {"oC", "oBlank", "oA"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(
            at::equal(result[0].features, expected_features({{2, {1, 2, 3}}, {2, {11, 12, 13}}})));
    CATCH_CHECK(result[0].read_ids_left == Ids{"iA", "iB", "iC"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"oA", "oBlank", "oC"});
}

CATCH_TEST_CASE(
        "EncoderReadAlignment::merge_adjacent_samples: empty right ID in 1st chunk matches an "
        "unmatched row of the 2nd chunk",
        TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // Row 1 of the first chunk is empty on the right, so the new read N from the second chunk
    // takes this row. No extra depth is needed.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {1, 2, 3}, {"iA", "iB", "iC"}, {"A", "__blank", "C"}),
            make_read_level_sample(2, 2, {13, 20, 11}, {"C", "N", "A"}, {"oC", "oN", "oA"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(
            at::equal(result[0].features, expected_features({{2, {1, 2, 3}}, {2, {11, 20, 13}}})));
    CATCH_CHECK(result[0].read_ids_left == Ids{"iA", "iB", "iC"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"oA", "oN", "oC"});
}

// Read IDs of rows which have no read (placeholders produced by the encoders, e.g. "__blank").
// Such IDs are "empty" and can be paired with any otherwise unmatched read ID in the other chunk.
// Non-empty IDs which do not match can never be paired.
CATCH_TEST_CASE(
        "EncoderReadAlignment::merge_adjacent_samples: left IDs of 2nd chunk are a subset of "
        "right IDs of 1st chunk plus new IDs",
        TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // Read B ends at the boundary and read N is new. Both are named, so they cannot be paired
    // and each needs its own row: depth = 4. Matching reads (A, C) must land in their rows.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {1, 2, 3}, {"iA", "iB", "iC"}, {"A", "B", "C"}),
            make_read_level_sample(2, 2, {13, 20, 11}, {"C", "N", "A"}, {"oC", "oN", "oA"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(at::equal(result[0].features,
                          expected_features({{2, {1, 2, 3, 0}}, {2, {11, 0, 13, 20}}})));
    // Padding rows are labelled "__blank".
    CATCH_CHECK(result[0].read_ids_left == Ids{"iA", "iB", "iC", "__blank"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"oA", "__blank", "oC", "oN"});
}

CATCH_TEST_CASE(
        "EncoderReadAlignment::merge_adjacent_samples: unmatched named IDs are not paired and "
        "both chunks are padded",
        TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // B (only in chunk 1) and X, Y (only in chunk 2) are all named. None may share a row,
    // so depth = A, B, C + X, Y = 5. Chunk 1 is padded with zero rows at the end, chunk 2
    // has a zero row where B lives.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {1, 2, 3}, {"iA", "iB", "iC"}, {"A", "B", "C"}),
            make_read_level_sample(2, 2, {13, 24, 25, 11}, {"C", "X", "Y", "A"},
                                   {"oC", "oX", "oY", "oA"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK(result[0].features.size(1) == 5);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(at::equal(result[0].features,
                          expected_features({{2, {1, 2, 3, 0, 0}}, {2, {11, 0, 13, 24, 25}}})));
    // Padding rows are labelled "__blank".
    CATCH_CHECK(result[0].read_ids_left == Ids{"iA", "iB", "iC", "__blank", "__blank"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"oA", "__blank", "oC", "oX", "oY"});
}

CATCH_TEST_CASE(
        "EncoderReadAlignment::merge_adjacent_samples: mix of matched, empty and unmatched IDs",
        TEST_GROUP) {
    const auto temp_dir = make_temp_dir("encoder_read_aln_merge_test");
    const auto encoder = make_merge_test_encoder(temp_dir.m_path);

    // Chunk 1 right: A (matches), B (named, only in chunk 1), empty.
    // Chunk 2 left:  N (named, only in chunk 2), A (matches), empty.
    // The empty chunk 2 ID takes B, the empty chunk 1 ID takes N. Depth stays 3.
    const auto result = encoder.merge_adjacent_samples({
            make_read_level_sample(0, 2, {1, 2, 3}, {"iA", "iB", "iC"}, {"A", "B", "__blank"}),
            make_read_level_sample(2, 2, {20, 11, 12}, {"N", "A", "__blank"},
                                   {"oN", "oA", "oBlank"}),
    });

    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK(result[0].features.size(1) == 3);
    CATCH_CHECK_NOTHROW(result[0].validate());
    CATCH_CHECK(
            at::equal(result[0].features, expected_features({{2, {1, 2, 3}}, {2, {11, 12, 20}}})));
    CATCH_CHECK(result[0].read_ids_left == Ids{"iA", "iB", "iC"});
    CATCH_CHECK(result[0].read_ids_right == Ids{"oA", "oBlank", "oN"});
}

}  // namespace dorado::secondary::tests
