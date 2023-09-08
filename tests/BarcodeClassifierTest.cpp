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
                CHECK(res.top_barcode_pos.second < seqlen);
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
                CHECK(res.top_barcode_pos.second < seqlen);
                CHECK(res.bottom_barcode_pos.first >= 0);
                CHECK(res.bottom_barcode_pos.second < seqlen);
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
                CHECK(res.top_barcode_pos.second < seqlen);
                CHECK(res.bottom_barcode_pos.first >= 0);
                CHECK(res.bottom_barcode_pos.second < seqlen);
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

TEST_CASE("BarcodeClassifierNode: check correct output files are created", TEST_GROUP) {
    using Catch::Matchers::Equals;

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    std::vector<std::string> kits = {"SQK-RPB004"};
    bool barcode_both_ends = GENERATE(true, false);
    bool no_trim = GENERATE(true, false);
    auto classifier = pipeline_desc.add_node<BarcodeClassifierNode>({sink}, 8, kits,
                                                                    barcode_both_ends, no_trim);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    auto read = dorado::ReadPtr::make();
    read->seq = "AAAA";
    read->qstring = "!!!!";
    read->read_id = "read_id";
    auto records = read->extract_sam_lines(false);

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

    for (auto& message : messages) {
        if (std::holds_alternative<BamPtr>(message)) {
            auto read = std::get<BamPtr>(std::move(message));
            bam1_t* rec = read.get();
            CHECK_THAT(bam_aux2Z(bam_aux_get(rec, "BC")), Equals("unclassified"));
        } else if (std::holds_alternative<ReadPtr>(message)) {
            auto read = std::get<ReadPtr>(std::move(message));
            CHECK(read->barcode == "unclassified");
        }
    }
}
