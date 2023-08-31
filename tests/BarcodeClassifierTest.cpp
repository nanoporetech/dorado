#include "read_pipeline/BarcodeClassifier.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"
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

    auto barcoding_kits = demux::barcode_kits_list_str();

    std::string s;
    std::stringstream ss(barcoding_kits);

    std::vector<std::string> kit_names;
    while (std::getline(ss, s, ' ')) {
        kit_names.push_back(s);
    }

    CHECK(kit_names.size() > 0);

    for (auto& kit_name : kit_names) {
        CHECK_NOTHROW(demux::BarcodeClassifier({kit_name}));
    }

    CHECK_NOTHROW(demux::BarcodeClassifier(kit_names));
}

TEST_CASE("BarcodeClassifier: instantiate barcode with unknown kit", TEST_GROUP) {
    CHECK_THROWS(demux::BarcodeClassifier({"MY_RANDOM_KIT"}));
}

TEST_CASE("BarcodeClassifier: test single ended barcode", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::BarcodeClassifier classifier({"SQK-RBK114-96"});

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
            }
        }
    }
}

TEST_CASE("BarcodeClassifier: test double ended barcode", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end"));

    demux::BarcodeClassifier classifier({"SQK-RPB004"});

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
            }
        }
    }
}

TEST_CASE("BarcodeClassifier: test double ended barcode with different variants", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/double_end_variant"));

    demux::BarcodeClassifier classifier({"EXP-PBC096"});

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
            }
        }
    }
}
