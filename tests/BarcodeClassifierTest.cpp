#include "read_pipeline/BarcodeClassifier.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

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
        CHECK_NOTHROW(demux::BarcodeClassifier({kit_name}, false));
    }

    CHECK_NOTHROW(demux::BarcodeClassifier(kit_names, false));
}

TEST_CASE("BarcodeClassifier: instantiate barcode with unknown kit", TEST_GROUP) {
    CHECK_THROWS(demux::BarcodeClassifier({"MY_RANDOM_KIT"}, false));
}

TEST_CASE("BarcodeClassifier: test single ended barcode", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::BarcodeClassifier classifier({"SQK-RBK114-96"}, false);

    for (std::string bc :
         {"SQK-RBK114-96_BC01", "SQK-RBK114-96_RBK39", "SQK-RBK114-96_BC92", "unclassified"}) {
        auto bc_file = data_dir / (bc + ".fastq");
        HtsReader reader(bc_file.string());
        while (reader.read()) {
            auto seqlen = reader.record->core.l_qseq;
            auto bseq = bam_get_seq(reader.record);
            std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
            auto res = classifier.barcode(seq);
            if (res.adapter_name == "unclassified") {
                CHECK(bc == res.adapter_name);
            } else {
                CHECK(bc == (res.kit + "_" + res.adapter_name));
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

    demux::BarcodeClassifier classifier({"SQK-RPB004"}, false);

    for (std::string bc :
         {"SQK-RPB004_BC01", "SQK-RPB004_BC05", "SQK-RPB004_BC11", "unclassified"}) {
        auto bc_file = data_dir / (bc + ".fastq");
        HtsReader reader(bc_file.string());
        while (reader.read()) {
            auto seqlen = reader.record->core.l_qseq;
            auto bseq = bam_get_seq(reader.record);
            std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
            auto res = classifier.barcode(seq);
            if (res.adapter_name == "unclassified") {
                CHECK(bc == res.adapter_name);
            } else {
                CHECK(bc == (res.kit + "_" + res.adapter_name));
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

    demux::BarcodeClassifier classifier({"EXP-PBC096"}, false);

    for (std::string bc :
         {"EXP-PBC096_BC04", "EXP-PBC096_BC37", "EXP-PBC096_BC83", "unclassified"}) {
        auto bc_file = data_dir / (bc + ".fastq");
        HtsReader reader(bc_file.string());
        while (reader.read()) {
            auto seqlen = reader.record->core.l_qseq;
            auto bseq = bam_get_seq(reader.record);
            std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
            auto res = classifier.barcode(seq);
            if (res.adapter_name == "unclassified") {
                CHECK(bc == res.adapter_name);
            } else {
                CHECK(bc == (res.kit + "_" + res.adapter_name));
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

    demux::BarcodeClassifier single_end_classifier({"EXP-PBC096"}, false);
    demux::BarcodeClassifier double_end_classifier({"EXP-PBC096"}, true);

    // Check case where both ends don't match.
    auto bc_file = data_dir / "EXP-PBC096_barcode_both_ends_fail.fastq";
    HtsReader reader(bc_file.string());
    while (reader.read()) {
        auto seqlen = reader.record->core.l_qseq;
        auto bseq = bam_get_seq(reader.record);
        std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
        auto single_end_res = single_end_classifier.barcode(seq);
        auto double_end_res = double_end_classifier.barcode(seq);
        CHECK(double_end_res.adapter_name == "unclassified");
        CHECK(single_end_res.adapter_name == "BC15");
    }
}

TEST_CASE("BarcodeClassifier: check barcodes on both ends - passing case", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end_variant"));

    demux::BarcodeClassifier single_end_classifier({"EXP-PBC096"}, false);
    demux::BarcodeClassifier double_end_classifier({"EXP-PBC096"}, true);
    // Check case where both ends do match.
    auto bc_file = data_dir / "EXP-PBC096_barcode_both_ends_pass.fastq";
    HtsReader reader(bc_file.string());
    while (reader.read()) {
        auto seqlen = reader.record->core.l_qseq;
        auto bseq = bam_get_seq(reader.record);
        std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
        auto single_end_res = single_end_classifier.barcode(seq);
        auto double_end_res = double_end_classifier.barcode(seq);
        CHECK(double_end_res.adapter_name == single_end_res.adapter_name);
        CHECK(single_end_res.adapter_name == "BC01");
    }
}

TEST_CASE("BarcodeClassifierNode: check read messages are correctly upadted after barcoding",
          TEST_GROUP) {
    using Catch::Matchers::Equals;

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    std::vector<std::string> kits = {"SQK-RPB004"};
    bool barcode_both_ends = GENERATE(true, false);
    bool no_trim = false;
    auto classifier = pipeline_desc.add_node<BarcodeClassifierNode>({sink}, 8, kits,
                                                                    barcode_both_ends, no_trim);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    // Create new read that is barcode - 100 As - barcode.
    auto read = dorado::ReadPtr::make();
    const std::string seq = std::string(100, 'A');
    const auto& kit_info_map = barcode_kits::get_kit_infos();
    const auto& barcodes = barcode_kits::get_barcodes();
    const std::string front_flank = kit_info_map.at("SQK-RPB004").top_front_flank +
                                    barcodes.at("BC01") +
                                    kit_info_map.at("SQK-RPB004").top_rear_flank;
    const std::string rear_flank = dorado::utils::reverse_complement(front_flank);
    const int stride = 6;
    read->seq = front_flank + seq + rear_flank;
    read->qstring = std::string(read->seq.length(), '!');
    read->read_id = "read_id";
    read->model_stride = stride;
    std::vector<uint8_t> moves;
    for (int i = 0; i < read->seq.length(); i++) {
        moves.push_back(1);
        moves.push_back(0);
    }
    read->moves = moves;

    // Generate mod prob table so only the first A after the front flank has a mod.
    const std::string mod_alphabet = "AXCGT";
    read->mod_base_info = std::make_shared<dorado::ModBaseInfo>(mod_alphabet, "6mA", "");
    read->base_mod_probs = std::vector<uint8_t>(read->seq.length() * mod_alphabet.size(), 0);

    for (int i = 0; i < read->seq.size(); i++) {
        switch (read->seq[i]) {
        case 'A':
            read->base_mod_probs[i * mod_alphabet.size()] = 255;
            break;
        case 'C':
            read->base_mod_probs[i * mod_alphabet.size() + 2] = 255;
            break;
        case 'G':
            read->base_mod_probs[i * mod_alphabet.size() + 3] = 255;
            break;
        case 'T':
            read->base_mod_probs[i * mod_alphabet.size() + 4] = 255;
            break;
        }
    }
    read->base_mod_probs[front_flank.length() * mod_alphabet.size()] = 20;         // A
    read->base_mod_probs[(front_flank.length() * mod_alphabet.size()) + 1] = 235;  // 6mA

    read->num_trimmed_samples = 0;

    auto records = read->extract_sam_lines(true /* emit moves*/, 10);

    // Push a Read type.
    pipeline->push_message(std::move(read));
    // Push BamPtr type.
    for (auto& rec : records) {
        pipeline->push_message(std::move(rec));
    }
    dorado::ReadPair dummy_read_pair;
    // Push a type not used by the ClassifierNode.
    pipeline->push_message(std::move(dummy_read_pair));

    pipeline->terminate(DefaultFlushOptions());

    CHECK(messages.size() == 3);

    const std::string expected_bc = "SQK-RPB004_BC01";
    std::vector<uint8_t> expected_move_vals;
    for (int i = 0; i < seq.length(); i++) {
        expected_move_vals.push_back(1);
        expected_move_vals.push_back(0);
    }
    const int additional_trimmed_samples =
            stride * 2 * front_flank.length();  // * 2 is because we have 2 moves per base

    for (auto& message : messages) {
        if (std::holds_alternative<BamPtr>(message)) {
            // Check trimming on the bam1_t struct.
            auto read = std::get<BamPtr>(std::move(message));
            bam1_t* rec = read.get();

            CHECK_THAT(bam_aux2Z(bam_aux_get(rec, "BC")), Equals(expected_bc));

            auto seq = dorado::utils::extract_sequence(rec, rec->core.l_qseq);
            CHECK(seq == seq);

            auto qual = dorado::utils::extract_quality(rec, rec->core.l_qseq);
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
        } else if (std::holds_alternative<ReadPtr>(message)) {
            // Check trimming on the Read type.
            auto read = std::get<ReadPtr>(std::move(message));

            CHECK(read->barcode == expected_bc);

            CHECK(read->seq == seq);

            CHECK(read->moves == expected_move_vals);

            // The mod probabilities table should not start mod at the first base.
            CHECK(read->base_mod_probs.size() == read->seq.size() * mod_alphabet.size());
            CHECK(read->base_mod_probs[0] == 20);
            CHECK(read->base_mod_probs[1] == 235);

            CHECK(read->num_trimmed_samples == additional_trimmed_samples);

            auto bams = read->extract_sam_lines(0, 10);
            auto& rec = bams[0];
            auto [mod_str, mod_probs] = dorado::utils::extract_modbase_info(rec.get());
        }
    }
}
