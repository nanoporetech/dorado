#include "demux/BarcodeClassifier.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/sequence_utils.h"

#include <ATen/Functions.h>
#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#define TEST_GROUP "[barcode_demux]"
namespace fs = std::filesystem;

using namespace dorado;

TEST_CASE("BarcodeClassifier: check instantiation for all kits", TEST_GROUP) {
    using Catch::Matchers::Contains;

    auto barcoding_kits = barcode_kits::barcode_kits_list_str();

    std::string s;
    std::stringstream ss(barcoding_kits);

    std::vector<std::string> kit_names;
    while (std::getline(ss, s, ' ')) {
        kit_names.push_back(s);
    }

    CHECK(kit_names.size() > 0);

    for (auto& kit_name : kit_names) {
        CHECK_NOTHROW(demux::BarcodeClassifier({kit_name}, std::nullopt, std::nullopt));
    }

    CHECK_NOTHROW(demux::BarcodeClassifier(kit_names, std::nullopt, std::nullopt));
}

TEST_CASE("BarcodeClassifier: instantiate barcode with unknown kit", TEST_GROUP) {
    CHECK_THROWS(demux::BarcodeClassifier({"MY_RANDOM_KIT"}, std::nullopt, std::nullopt));
}

TEST_CASE("BarcodeClassifier: test single ended barcode", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::BarcodeClassifier classifier({"SQK-RBK114-96"}, std::nullopt, std::nullopt);

    for (std::string bc :
         {"SQK-RBK114-96_BC01", "SQK-RBK114-96_RBK39", "SQK-RBK114-96_BC92", "unclassified"}) {
        auto bc_file = data_dir / (bc + ".fastq");
        HtsReader reader(bc_file.string(), std::nullopt);
        while (reader.read()) {
            auto seqlen = reader.record->core.l_qseq;
            std::string seq = utils::extract_sequence(reader.record.get());
            auto res = classifier.barcode(seq, false, std::nullopt);
            if (res.barcode_name == "unclassified") {
                CHECK(bc == res.barcode_name);
            } else {
                CHECK(bc == (res.kit + "_" + res.barcode_name));
                CHECK(res.top_barcode_pos.second > res.top_barcode_pos.first);
                CHECK(res.top_barcode_pos.first >= 0);
                CHECK(res.top_barcode_pos.second <= seqlen);
                CHECK(res.bottom_barcode_pos.first == -1);
                CHECK(res.bottom_barcode_pos.second == -1);
            }
        }
    }
}

TEST_CASE("BarcodeClassifier: test double ended barcode", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end"));

    demux::BarcodeClassifier classifier({"SQK-RPB004"}, std::nullopt, std::nullopt);

    for (std::string bc :
         {"SQK-RPB004_BC01", "SQK-RPB004_BC05", "SQK-RPB004_BC11", "unclassified"}) {
        auto bc_file = data_dir / (bc + ".fastq");
        HtsReader reader(bc_file.string(), std::nullopt);
        while (reader.read()) {
            auto seqlen = reader.record->core.l_qseq;
            std::string seq = utils::extract_sequence(reader.record.get());
            auto res = classifier.barcode(seq, false, std::nullopt);
            if (res.barcode_name == "unclassified") {
                CHECK(bc == res.barcode_name);
            } else {
                CHECK(bc == (res.kit + "_" + res.barcode_name));
                CHECK(res.top_barcode_pos.second > res.top_barcode_pos.first);
                CHECK(res.bottom_barcode_pos.second > res.bottom_barcode_pos.first);
                CHECK(res.top_barcode_pos.first >= 0);
                CHECK(res.top_barcode_pos.second <= seqlen);
                CHECK(res.bottom_barcode_pos.first >= 0);
                CHECK(res.bottom_barcode_pos.second <= seqlen);
                CHECK(res.top_barcode_pos.second < res.bottom_barcode_pos.first);
            }
        }
    }
}

TEST_CASE("BarcodeClassifier: test double ended barcode with different variants", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end_variant"));

    demux::BarcodeClassifier classifier({"EXP-PBC096"}, std::nullopt, std::nullopt);

    for (std::string bc :
         {"EXP-PBC096_BC04", "EXP-PBC096_BC37", "EXP-PBC096_BC83", "unclassified"}) {
        auto bc_file = data_dir / (bc + ".fastq");
        HtsReader reader(bc_file.string(), std::nullopt);
        while (reader.read()) {
            auto seqlen = reader.record->core.l_qseq;
            std::string seq = utils::extract_sequence(reader.record.get());
            auto res = classifier.barcode(seq, false, std::nullopt);
            if (res.barcode_name == "unclassified") {
                CHECK(bc == res.barcode_name);
            } else {
                CHECK(bc == (res.kit + "_" + res.barcode_name));
                CHECK(res.top_barcode_pos.second > res.top_barcode_pos.first);
                CHECK(res.bottom_barcode_pos.second > res.bottom_barcode_pos.first);
                CHECK(res.top_barcode_pos.first >= 0);
                CHECK(res.top_barcode_pos.second <= seqlen);
                CHECK(res.bottom_barcode_pos.first >= 0);
                CHECK(res.bottom_barcode_pos.second <= seqlen);
                CHECK(res.top_barcode_pos.second < res.bottom_barcode_pos.first);
            }
        }
    }
}

TEST_CASE("BarcodeClassifier: check barcodes on both ends - failing case", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end_variant"));

    demux::BarcodeClassifier classifier({"EXP-PBC096"}, std::nullopt, std::nullopt);

    // Check case where both ends don't match.
    auto bc_file = data_dir / "EXP-PBC096_barcode_both_ends_fail.fastq";
    HtsReader reader(bc_file.string(), std::nullopt);
    while (reader.read()) {
        std::string seq = utils::extract_sequence(reader.record.get());
        auto single_end_res = classifier.barcode(seq, false, std::nullopt);
        auto double_end_res = classifier.barcode(seq, true, std::nullopt);
        CHECK(double_end_res.barcode_name == "unclassified");
        CHECK(single_end_res.barcode_name == "BC01");
    }
}

TEST_CASE("BarcodeClassifier: check barcodes on both ends - passing case", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end_variant"));

    demux::BarcodeClassifier classifier({"EXP-PBC096"}, std::nullopt, std::nullopt);

    // Check case where both ends do match.
    auto bc_file = data_dir / "EXP-PBC096_barcode_both_ends_pass.fastq";
    HtsReader reader(bc_file.string(), std::nullopt);
    while (reader.read()) {
        std::string seq = utils::extract_sequence(reader.record.get());
        auto single_end_res = classifier.barcode(seq, false, std::nullopt);
        auto double_end_res = classifier.barcode(seq, true, std::nullopt);
        CHECK(double_end_res.barcode_name == single_end_res.barcode_name);
        CHECK(single_end_res.barcode_name == "BC01");
    }
}

TEST_CASE("BarcodeClassifier: check presence of midstrand barcode double ended kit", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end_variant"));

    demux::BarcodeClassifier classifier({"EXP-PBC096"}, std::nullopt, std::nullopt);

    // Check case where both ends do match.
    auto bc_file = data_dir / "EXP-PBC096_midstrand.fasta";
    HtsReader reader(bc_file.string(), std::nullopt);
    while (reader.read()) {
        std::string seq = utils::extract_sequence(reader.record.get());
        auto res = classifier.barcode(seq, false, std::nullopt);
        CHECK(res.barcode_name == "unclassified");
        CHECK(res.found_midstrand);
    }
}

TEST_CASE("BarcodeClassifier: check presence of midstrand barcode single ended kit", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::BarcodeClassifier classifier({"SQK-RBK114-96"}, std::nullopt, std::nullopt);

    // Check case where both ends do match.
    auto bc_file = data_dir / "SQK-RBK114-96_midstrand.fasta";
    HtsReader reader(bc_file.string(), std::nullopt);
    while (reader.read()) {
        std::string seq = utils::extract_sequence(reader.record.get());
        auto res = classifier.barcode(seq, false, std::nullopt);
        CHECK(res.barcode_name == "unclassified");
        CHECK(res.found_midstrand);
    }
}

TEST_CASE(
        "BarcodeClassifierNode: check read messages are correctly updated after classification and "
        "trimming",
        TEST_GROUP) {
    using Catch::Matchers::Equals;

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    std::vector<std::string> kits = {"SQK-RPB004"};
    const bool barcode_both_ends = GENERATE(true, false);
    const bool use_per_read_barcoding = GENERATE(true, false);
    CAPTURE(barcode_both_ends);
    CAPTURE(use_per_read_barcoding);
    constexpr bool no_trim = false;
    auto barcoding_info = dorado::create_barcoding_info(kits, barcode_both_ends, !no_trim,
                                                        std::nullopt, std::nullopt, std::nullopt);
    if (use_per_read_barcoding) {
        pipeline_desc.add_node<BarcodeClassifierNode>({sink}, 8);
    } else {
        pipeline_desc.add_node<BarcodeClassifierNode>({sink}, 8, kits, barcode_both_ends, no_trim,
                                                      std::nullopt, std::nullopt, std::nullopt);
    }

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    // Create new read that is barcode - 100 As - barcode.
    auto read = std::make_unique<SimplexRead>();
    const std::string nonbc_seq = std::string(100, 'A');
    const auto& kit_info_map = barcode_kits::get_kit_infos();
    const auto& barcodes = barcode_kits::get_barcodes();
    const std::string front_flank = kit_info_map.at("SQK-RPB004").top_front_flank +
                                    barcodes.at("BC01") +
                                    kit_info_map.at("SQK-RPB004").top_rear_flank;
    const std::string rear_flank = dorado::utils::reverse_complement(front_flank);
    const int stride = 6;
    read->read_common.seq = front_flank + nonbc_seq + rear_flank;
    read->read_common.qstring = std::string(read->read_common.seq.length(), '!');
    read->read_common.read_id = "read_id";
    read->read_common.model_stride = stride;

    if (use_per_read_barcoding) {
        read->read_common.barcoding_info = barcoding_info;
    }

    std::vector<uint8_t> moves;
    for (size_t i = 0; i < read->read_common.seq.length(); i++) {
        moves.push_back(1);
        moves.push_back(0);
    }
    read->read_common.moves = moves;
    read->read_common.raw_data = at::zeros(moves.size() * stride);

    // Generate mod prob table so only the first A after the front flank has a mod.
    const std::vector<std::string> mod_alphabet = {"A", "a", "C", "G", "T"};
    read->read_common.mod_base_info =
            std::make_shared<dorado::ModBaseInfo>(mod_alphabet, "6mA", "");
    read->read_common.base_mod_probs =
            std::vector<uint8_t>(read->read_common.seq.length() * mod_alphabet.size(), 0);

    for (size_t i = 0; i < read->read_common.seq.size(); i++) {
        switch (read->read_common.seq[i]) {
        case 'A':
            read->read_common.base_mod_probs[i * mod_alphabet.size()] = 255;
            break;
        case 'C':
            read->read_common.base_mod_probs[i * mod_alphabet.size() + 2] = 255;
            break;
        case 'G':
            read->read_common.base_mod_probs[i * mod_alphabet.size() + 3] = 255;
            break;
        case 'T':
            read->read_common.base_mod_probs[i * mod_alphabet.size() + 4] = 255;
            break;
        }
    }
    read->read_common.base_mod_probs[front_flank.length() * mod_alphabet.size()] = 20;  // A
    read->read_common.base_mod_probs[(front_flank.length() * mod_alphabet.size()) + 1] =
            235;  // 6mA

    read->read_common.num_trimmed_samples = 0;

    auto records = read->read_common.extract_sam_lines(true /* emit moves*/, 10, false);

    // Push a Read type.
    pipeline->push_message(std::move(read));
    // Push BamPtr type.
    if (!use_per_read_barcoding) {
        // The classifier node will only barcode Simplex reads so BamPtr is not relevant here
        for (auto& rec : records) {
            pipeline->push_message(std::move(rec));
        }
    }
    dorado::ReadPair dummy_read_pair;
    // Push a type not used by the ClassifierNode.
    pipeline->push_message(std::move(dummy_read_pair));

    pipeline->terminate(DefaultFlushOptions());

    const size_t num_expected_messages = use_per_read_barcoding ? 2 : 3;
    CHECK(messages.size() == num_expected_messages);

    const std::string expected_bc = "SQK-RPB004_barcode01";
    std::vector<uint8_t> expected_move_vals;
    for (size_t i = 0; i < nonbc_seq.length(); i++) {
        expected_move_vals.push_back(1);
        expected_move_vals.push_back(0);
    }
    const int additional_trimmed_samples =
            int(stride * 2 * front_flank.length());  // * 2 is because we have 2 moves per base

    for (auto& message : messages) {
        if (std::holds_alternative<BamPtr>(message)) {
            // Check trimming on the bam1_t struct.
            auto msg_read = std::get<BamPtr>(std::move(message));
            bam1_t* rec = msg_read.get();

            CHECK_THAT(bam_aux2Z(bam_aux_get(rec, "BC")), Equals(expected_bc));

            auto seq = dorado::utils::extract_sequence(rec);
            CHECK(nonbc_seq == seq);

            auto qual = dorado::utils::extract_quality(rec);
            CHECK(qual.size() == seq.length());

            auto [_, move_vals] = dorado::utils::extract_move_table(rec);
            CHECK(move_vals == expected_move_vals);

            // The mod should now be at the very first base.
            const std::string expected_mod_str = "A+a.,0;";
            const std::vector<uint8_t> expected_mod_probs = {235};
            auto [mod_str, mod_probs] = dorado::utils::extract_modbase_info(rec);
            CHECK(mod_str == expected_mod_str);
            CHECK_THAT(mod_probs, Equals(std::vector<uint8_t>{235}));

            CHECK(bam_aux2i(bam_aux_get(rec, "ts")) == additional_trimmed_samples);
            // 100 is trimmed sequence length. 51 is leading flank length.
            // * 2 is number of moves per base.
            // ns is shorter than the number of raw samples because the signal
            // corresponding to the trimmed trailing flank needs to be removed from the end
            // of the original signal.
            CHECK(bam_aux2i(bam_aux_get(rec, "ns")) == ((100 + 51) * 2 * stride));
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            // Check trimming on the Read type.
            auto msg_read = std::get<SimplexReadPtr>(std::move(message));
            const ReadCommon& read_common = msg_read->read_common;

            CHECK(read_common.barcode == expected_bc);

            CHECK(read_common.seq == nonbc_seq);

            CHECK(read_common.moves == expected_move_vals);

            // The mod probabilities table should not start mod at the first base.
            CHECK(read_common.base_mod_probs.size() ==
                  read_common.seq.size() * mod_alphabet.size());
            CHECK(read_common.base_mod_probs[0] == 20);
            CHECK(read_common.base_mod_probs[1] == 235);

            CHECK(read_common.num_trimmed_samples == uint64_t(additional_trimmed_samples));
            // Number of trimmed bases is 100, so number of moves should be 2 * 100.
            CHECK(read_common.get_raw_data_samples() == 100 * 2 * stride);

            auto bams = read_common.extract_sam_lines(0, 10, false);
            auto& rec = bams[0];
            auto [mod_str, mod_probs] = dorado::utils::extract_modbase_info(rec.get());
        }
    }
}

TEST_CASE("BarcodeClassifierNode: test for proper trimming and alignment data stripping",
          TEST_GROUP) {
    using Catch::Matchers::Equals;

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    std::vector<std::string> kits = {"SQK-16S024"};
    bool barcode_both_ends = false;
    bool no_trim = false;
    pipeline_desc.add_node<BarcodeClassifierNode>({sink}, 8, kits, barcode_both_ends, no_trim,
                                                  std::nullopt, std::nullopt, std::nullopt);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);
    fs::path data_dir = fs::path(get_data_dir("barcode_demux"));
    auto bc_file = data_dir / "simple_mapped_reads.sam";

    // First read should be unclassified.
    HtsReader reader(bc_file.string(), std::nullopt);
    reader.read();
    BamPtr read1(bam_dup1(reader.record.get()));
    std::string id_in1 = bam_get_qname(read1.get());
    auto orig_seq1 = dorado::utils::extract_sequence(read1.get());
    pipeline->push_message(std::move(read1));

    // Second read should be classified.
    reader.read();
    BamPtr read2(bam_dup1(reader.record.get()));
    std::string id_in2 = bam_get_qname(read2.get());
    auto orig_seq2 = dorado::utils::extract_sequence(read2.get());
    pipeline->push_message(std::move(read2));

    pipeline->terminate(DefaultFlushOptions());

    CHECK(messages.size() == 2);

    read1 = std::get<BamPtr>(std::move(messages[0]));
    read2 = std::get<BamPtr>(std::move(messages[1]));

    // Reads may not come back in the same order.
    std::string id_out1 = bam_get_qname(read1.get());
    std::string id_out2 = bam_get_qname(read2.get());
    if (id_out1 != id_in1) {
        read1.swap(read2);
    }

    // First read should be unclassified and untrimmed.
    auto seq1 = dorado::utils::extract_sequence(read1.get());
    CHECK(seq1 == orig_seq1);
    CHECK_THAT(bam_aux2Z(bam_aux_get(read1.get(), "BC")), Equals("unclassified"));

    // Second read should be classified and trimmed.
    auto seq2 = dorado::utils::extract_sequence(read2.get());
    CHECK(seq2 ==
          seq1);  // Sequence 2 is just sequence 1 plus the barcode, which should be trimmed.
    CHECK_THAT(bam_aux2Z(bam_aux_get(read2.get(), "BC")), Equals("SQK-16S024_barcode01"));

    // Check to make sure alignment data has been stripped from both reads.
    CHECK(read1->core.tid == -1);
    CHECK(bam_aux_get(read1.get(), "bh") == nullptr);
    CHECK(read2->core.tid == -1);
    CHECK(bam_aux_get(read2.get(), "bh") == nullptr);
}

struct CustomDoubleEndedKitInput {
    std::string kit_file;
    std::optional<std::string> seqs_file;
};

TEST_CASE("BarcodeClassifier: test custom kit with double ended barcode", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end"));
    auto [kit_file, seqs_file] = GENERATE(
            CustomDoubleEndedKitInput{
                    (fs::path(get_data_dir("barcode_demux/custom_barcodes")) / "RPB004.toml")
                            .string(),
                    std::nullopt},
            CustomDoubleEndedKitInput{
                    (fs::path(get_data_dir("barcode_demux/custom_barcodes")) / "RPB004.toml")
                            .string(),
                    (fs::path(get_data_dir("barcode_demux/custom_barcodes")) /
                     "RPB004_sequences.fasta")
                            .string()});

    demux::BarcodeClassifier classifier({}, kit_file, seqs_file);

    for (std::string bc :
         {"SQK-RPB004_BC01", "SQK-RPB004_BC05", "SQK-RPB004_BC11", "unclassified"}) {
        auto bc_file = data_dir / (bc + ".fastq");
        HtsReader reader(bc_file.string(), std::nullopt);
        while (reader.read()) {
            auto seqlen = reader.record->core.l_qseq;
            std::string seq = utils::extract_sequence(reader.record.get());
            auto res = classifier.barcode(seq, false, std::nullopt);
            if (res.barcode_name == "unclassified") {
                CHECK(bc == res.barcode_name);
            } else {
                CHECK(bc == (res.kit + "_" + res.barcode_name));
                CHECK(res.top_barcode_pos.second > res.top_barcode_pos.first);
                CHECK(res.bottom_barcode_pos.second > res.bottom_barcode_pos.first);
                CHECK(res.top_barcode_pos.first >= 0);
                CHECK(res.top_barcode_pos.second <= seqlen);
                CHECK(res.bottom_barcode_pos.first >= 0);
                CHECK(res.bottom_barcode_pos.second <= seqlen);
                CHECK(res.top_barcode_pos.second < res.bottom_barcode_pos.first);
            }
        }
    }
}

TEST_CASE(
        "BarcodeClassifier: Fail if no kit name is passed and custom kit doesn't contain "
        "arrangement",
        TEST_GROUP) {
    const fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto kit_file = data_dir / "scoring_params.toml";

    CHECK_THROWS_WITH(
            demux::BarcodeClassifier({}, kit_file.string(), std::nullopt),
            "Either custom kit must include kit arrangement or a kit name needs to be passed in.");
}
