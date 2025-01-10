#include "demux/AdapterDetector.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "demux/Trimmer.h"
#include "demux/adapter_info.h"
#include "read_pipeline/AdapterDetectorNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/TrimmerNode.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <ATen/Functions.h>
#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#define TEST_GROUP "[adapter_detect]"

namespace {
const std::string TEST_KIT = "SQK-LSK114";

using Query = dorado::demux::AdapterDetector::Query;

void sort_queries(std::vector<Query>& queries) {
    auto query_lt = [](const Query& a, const Query& b) -> bool { return a.name < b.name; };
    std::sort(queries.begin(), queries.end(), query_lt);
}

}  // namespace

namespace fs = std::filesystem;

using namespace dorado;

TEST_CASE("AdapterDetector: test classification", TEST_GROUP) {
    dorado::AdapterScoreResult res;

    // Consistent 5' primer at front and 3' primer at rear
    res.front.name = "test_FWD_FRONT";
    res.rear.name = "test_FWD_REAR";
    auto classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == "test_FWD");
    CHECK(classification.orientation == 1);
    CHECK(classification.orientation_char() == '+');

    // Only 5' primer at front
    res.front.name = "test_FWD_FRONT";
    res.rear.name = UNCLASSIFIED;
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == "test_FWD");
    CHECK(classification.orientation == 1);
    CHECK(classification.orientation_char() == '+');

    // Only 3' primer at rear
    res.front.name = UNCLASSIFIED;
    res.rear.name = "test_FWD_REAR";
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == "test_FWD");
    CHECK(classification.orientation == 1);
    CHECK(classification.orientation_char() == '+');

    // Consistent 3' primer at front and 5' primer at rear
    res.front.name = "test_REV_FRONT";
    res.rear.name = "test_REV_REAR";
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == "test_REV");
    CHECK(classification.orientation == -1);
    CHECK(classification.orientation_char() == '-');

    // Only 3' primer at front
    res.front.name = "test_REV_FRONT";
    res.rear.name = UNCLASSIFIED;
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == "test_REV");
    CHECK(classification.orientation == -1);
    CHECK(classification.orientation_char() == '-');

    // Only 5' primer at rear
    res.front.name = UNCLASSIFIED;
    res.rear.name = "test_REV_REAR";
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == "test_REV");
    CHECK(classification.orientation == -1);
    CHECK(classification.orientation_char() == '-');

    // No primers found at either end
    res.front.name = UNCLASSIFIED;
    res.rear.name = UNCLASSIFIED;
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == UNCLASSIFIED);
    CHECK(classification.orientation == 0);
    CHECK(classification.orientation_char() == '?');

    // Inconsistent 5' primer at front and 3' primer at rear
    res.front.name = "test1_FWD_FRONT";
    res.rear.name = "test2_FWD_REAR";
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == UNCLASSIFIED);
    CHECK(classification.orientation == 0);
    CHECK(classification.orientation_char() == '?');

    // 5' primer found at both front and rear
    res.front.name = "test1_FWD_FRONT";
    res.rear.name = "test1_REV_REAR";
    classification = demux::AdapterDetector::classify_primers(res);
    CHECK(classification.primer_name == UNCLASSIFIED);
    CHECK(classification.orientation == 0);
    CHECK(classification.orientation_char() == '?');
}

TEST_CASE("AdapterDetector: test adapter detection", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::AdapterDetector detector(std::nullopt);
    const auto& adapters = detector.get_adapter_sequences(TEST_KIT);

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < adapters.size(); ++i) {
        // First put the front adapter only at the beginning, with 6 bases in front of it.
        auto new_sequence1 = "ACGTAC" + adapters[i].front_sequence + seq;
        auto res1 = detector.find_adapters(new_sequence1, TEST_KIT);
        CHECK(res1.front.name == adapters[i].name + "_FRONT");
        CHECK(res1.front.position ==
              std::make_pair(6, int(adapters[i].front_sequence.length()) + 5));
        CHECK(res1.front.score == 1.0f);
        CHECK(res1.rear.score < 0.7f);

        // Now put the rear adapter at the end, with 3 bases after it.
        auto new_sequence2 = seq + adapters[i].rear_sequence + "TTT";
        auto res2 = detector.find_adapters(new_sequence2, TEST_KIT);
        CHECK(res2.front.score < 0.7f);
        CHECK(res2.rear.name == adapters[i].name + "_REAR");
        CHECK(res2.rear.position ==
              std::make_pair(int(seq.length()),
                             int(seq.length() + adapters[i].rear_sequence.length()) - 1));
        CHECK(res2.rear.score == 1.0f);

        // Now put them both in.
        auto new_sequence3 =
                "TGCA" + adapters[i].front_sequence + seq + adapters[i].rear_sequence + "GTA";
        auto res3 = detector.find_adapters(new_sequence3, TEST_KIT);
        CHECK(res3.front.name == adapters[i].name + "_FRONT");
        CHECK(res3.front.position ==
              std::make_pair(4, int(adapters[i].front_sequence.length()) + 3));
        CHECK(res3.front.score == 1.0f);
        CHECK(res3.rear.name == adapters[i].name + "_REAR");
        CHECK(res3.rear.position ==
              std::make_pair(int(adapters[i].front_sequence.length() + seq.length()) + 4,
                             int(adapters[i].front_sequence.length() + seq.length() +
                                 adapters[i].rear_sequence.length()) +
                                     3));
        CHECK(res3.rear.score == 1.0f);
    }
}

TEST_CASE("AdapterDetector: test primer detection", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::AdapterDetector detector(std::nullopt);
    auto primers = detector.get_primer_sequences(TEST_KIT);
    sort_queries(primers);

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    CHECK(primers.size() == 2);
    for (size_t i = 0; i < primers.size(); ++i) {
        auto new_sequence1 =
                "ACGTAC" + primers[i].front_sequence + seq + primers[i].rear_sequence + "TTT";
        auto res = detector.find_primers(new_sequence1, TEST_KIT);
        CHECK(res.front.name == primers[i].name + "_FRONT");
        CHECK(res.front.position == std::make_pair(6, int(primers[i].front_sequence.length()) + 5));
        CHECK(res.front.score == 1.0f);
        CHECK(res.rear.name == primers[i].name + "_REAR");
        CHECK(res.rear.position ==
              std::make_pair(int(primers[i].front_sequence.length() + seq.length()) + 6,
                             int(primers[i].front_sequence.length() + seq.length() +
                                 primers[i].rear_sequence.length()) +
                                     5));
        CHECK(res.rear.score == 1.0f);
        auto classification = demux::AdapterDetector::classify_primers(res);
        CHECK(classification.primer_name == primers[i].name);
        int expected_orientation = (i == 0) ? 1 : -1;
        CHECK(classification.orientation == expected_orientation);
    }
}

TEST_CASE("AdapterDetector: test custom primer detection with kit", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));
    fs::path seq_dir = fs::path(get_data_dir("adapter_trim"));
    auto custom_primer_file = (seq_dir / "custom_adapter_primer_with_kits.fasta").string();

    demux::AdapterDetector detector(custom_primer_file);
    const auto& adapters = detector.get_adapter_sequences("TEST_KIT1");
    auto primers = detector.get_primer_sequences("TEST_KIT2");
    sort_queries(primers);
    // Make sure the adapters and primers have been properly loaded.
    std::string expected_adapter_name = "adapter1";
    std::string expected_adapter_front = "AGGGAACT";
    std::string expected_adapter_rear = "AGTTCC";
    std::vector<std::string> expected_primer_names = {"primer1_FWD", "primer1_REV"};
    std::vector<std::string> expected_primer_fronts = {"TGCGAAT", "CAGAGGTC"};
    std::vector<std::string> expected_primer_rears = {"GACCTCTG", "ATTCGCA"};
    REQUIRE(adapters.size() == 1);
    CHECK(adapters[0].name == expected_adapter_name);
    CHECK(adapters[0].front_sequence == expected_adapter_front);
    CHECK(adapters[0].rear_sequence == expected_adapter_rear);

    REQUIRE(primers.size() == 2);
    CHECK(primers[0].name == expected_primer_names[0]);
    CHECK(primers[0].front_sequence == expected_primer_fronts[0]);
    CHECK(primers[0].rear_sequence == expected_primer_rears[0]);
    CHECK(primers[1].name == expected_primer_names[1]);
    CHECK(primers[1].front_sequence == expected_primer_fronts[1]);
    CHECK(primers[1].rear_sequence == expected_primer_rears[1]);

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < primers.size(); ++i) {
        // Put the front primer at the beginning, and the rear primer at the end.
        auto new_sequence1 =
                "ACGTAC" + primers[i].front_sequence + seq + primers[i].rear_sequence + "TTT";
        auto res = detector.find_primers(new_sequence1, "TEST_KIT2");
        CHECK(res.front.name == primers[i].name + "_FRONT");
        CHECK(res.front.position == std::make_pair(6, int(primers[i].front_sequence.length()) + 5));
        CHECK(res.front.score == 1.0f);
        CHECK(res.rear.name == primers[i].name + "_REAR");
        CHECK(res.rear.position ==
              std::make_pair(int(primers[i].front_sequence.length() + seq.length()) + 6,
                             int(primers[i].front_sequence.length() + seq.length() +
                                 primers[i].rear_sequence.length()) +
                                     5));
        CHECK(res.rear.score == 1.0f);
        auto classification = demux::AdapterDetector::classify_primers(res);
        CHECK(classification.primer_name == primers[i].name);
        int expected_orientation = (i == 0) ? 1 : -1;
        CHECK(classification.orientation == expected_orientation);
    }
}

TEST_CASE("AdapterDetector: test custom primer detection without kit", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));
    fs::path seq_dir = fs::path(get_data_dir("adapter_trim"));

    // The sequences in this file contain no kit specification, so they will be
    // treated as being suitable for all kits.
    auto custom_primer_file = (seq_dir / "custom_adapter_primer_no_kits.fasta").string();

    demux::AdapterDetector detector(custom_primer_file);
    const auto& adapters = detector.get_adapter_sequences("TEST_KIT1");
    auto primers = detector.get_primer_sequences("TEST_KIT2");
    sort_queries(primers);
    // Make sure the adapters and primers have been properly loaded.
    std::string expected_adapter_name = "adapter1";
    std::string expected_adapter_front = "AGGGAACT";
    std::string expected_adapter_rear = "AGTTCC";
    std::vector<std::string> expected_primer_names = {"primer1_FWD", "primer1_REV", "primer2_FWD",
                                                      "primer2_REV"};
    std::vector<std::string> expected_primers_front = {"TGCGAAT", "CAGAGGTC", "CTGACGT", "CAATGAA"};
    std::vector<std::string> expected_primers_rear = {"GACCTCTG", "ATTCGCA", "TTCATTG", "ACGTCAG"};
    CHECK(adapters.size() == 1);
    CHECK(primers.size() == 4);
    for (size_t i = 0; i < primers.size(); ++i) {
        CHECK(primers[i].name == expected_primer_names[i]);
        CHECK(primers[i].front_sequence == expected_primers_front[i]);
        CHECK(primers[i].rear_sequence == expected_primers_rear[i]);
    }
    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < primers.size(); ++i) {
        // Put the front primer at the beginning, and the rear primer at the end.
        auto new_sequence1 =
                "ACGTAC" + primers[i].front_sequence + seq + primers[i].rear_sequence + "TTT";
        auto res = detector.find_primers(new_sequence1, "TEST_KIT2");
        CHECK(res.front.name == primers[i].name + "_FRONT");
        CHECK(res.front.position == std::make_pair(6, int(primers[i].front_sequence.length()) + 5));
        CHECK(res.front.score == 1.0f);
        CHECK(res.rear.name == primers[i].name + "_REAR");
        CHECK(res.rear.position ==
              std::make_pair(int(primers[i].front_sequence.length() + seq.length()) + 6,
                             int(primers[i].front_sequence.length() + seq.length() +
                                 primers[i].rear_sequence.length()) +
                                     5));
        CHECK(res.rear.score == 1.0f);
        auto classification = demux::AdapterDetector::classify_primers(res);
        CHECK(classification.primer_name == primers[i].name);
        int expected_orientation = (i % 2 == 0) ? 1 : -1;
        CHECK(classification.orientation == expected_orientation);
    }
}

void detect_and_trim(SimplexRead& read) {
    demux::AdapterDetector detector(std::nullopt);
    auto seqlen = int(read.read_common.seq.length());
    std::pair<int, int> adapter_trim_interval = {0, seqlen};
    std::pair<int, int> primer_trim_interval = {0, seqlen};

    auto adapter_res = detector.find_adapters(read.read_common.seq, TEST_KIT);
    adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    auto primer_res = detector.find_primers(read.read_common.seq, TEST_KIT);
    primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
    std::pair<int, int> trim_interval = adapter_trim_interval;
    trim_interval.first = std::max(trim_interval.first, primer_trim_interval.first);
    trim_interval.second = std::min(trim_interval.second, primer_trim_interval.second);
    CHECK(trim_interval.first < trim_interval.second);
    Trimmer::trim_sequence(read, trim_interval);
    read.read_common.adapter_trim_interval = trim_interval;
}

TEST_CASE(
        "AdapterDetectorNode: check read messages are correctly updated after adapter/primer "
        "detection and trimming",
        TEST_GROUP) {
    using Catch::Matchers::Equals;

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto trimmer = pipeline_desc.add_node<TrimmerNode>({sink}, 1);
    pipeline_desc.add_node<AdapterDetectorNode>({trimmer}, 8);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    // Create new read that is [LSK110_FWD] - [cDNA_VNP_FWD] - 200 As - [cDNA_VNP_REV] [LSK110_REV].
    auto read = std::make_unique<SimplexRead>();
    const std::string nonbc_seq = std::string(200, 'A');
    demux::AdapterDetector detector(std::nullopt);
    const auto& adapters = detector.get_adapter_sequences(TEST_KIT);
    const auto& primers = detector.get_primer_sequences(TEST_KIT);
    const auto& front_adapter = adapters[0].front_sequence;
    const auto& front_primer = primers[0].front_sequence;
    const auto& rear_adapter = adapters[0].rear_sequence;
    const auto& rear_primer = primers[0].rear_sequence;
    const int stride = 6;
    read->read_common.seq = front_adapter + front_primer + nonbc_seq + rear_primer + rear_adapter;
    read->read_common.qstring = std::string(read->read_common.seq.length(), '!');
    read->read_common.read_id = "read_id";
    read->read_common.model_stride = stride;
    read->read_common.sequencing_kit = TEST_KIT;

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
    auto flank_size = front_adapter.length() + front_primer.length();
    read->read_common.base_mod_probs[flank_size * mod_alphabet.size()] = 20;         // A
    read->read_common.base_mod_probs[(flank_size * mod_alphabet.size()) + 1] = 235;  // 6mA
    read->read_common.num_trimmed_samples = 0;

    auto records = read->read_common.extract_sam_lines(true /* emit moves */, 10, false);
    BamPtr record_copy(bam_dup1(records[0].get()));

    auto client_info = std::make_shared<dorado::DefaultClientInfo>();
    client_info->contexts().register_context<const dorado::demux::AdapterInfo>(
            std::make_shared<const dorado::demux::AdapterInfo>(
                    dorado::demux::AdapterInfo{true, true, false, std::nullopt}));
    read->read_common.client_info = std::move(client_info);

    BamMessage bam_read{std::move(record_copy), read->read_common.client_info};
    bam_read.sequencing_kit = TEST_KIT;

    // Push a Symplex read type.
    pipeline->push_message(std::move(read));

    // Push a BAM read type.
    pipeline->push_message(std::move(bam_read));

    // Push a type not used by the ClassifierNode.
    dorado::ReadPair dummy_read_pair;
    pipeline->push_message(std::move(dummy_read_pair));

    pipeline->terminate(DefaultFlushOptions());

    const size_t num_expected_messages = 3;
    CHECK(messages.size() == num_expected_messages);

    std::vector<uint8_t> expected_move_vals;
    for (size_t i = 0; i < nonbc_seq.length(); i++) {
        expected_move_vals.push_back(1);
        expected_move_vals.push_back(0);
    }
    const int additional_trimmed_samples =
            int(stride * 2 * flank_size);  // * 2 is because we have 2 moves per base

    for (auto& message : messages) {
        if (std::holds_alternative<BamMessage>(message)) {
            auto bam_message = std::get<BamMessage>(std::move(message));
            bam1_t* rec = bam_message.bam_ptr.get();

            // Check trimming on the bam1_t struct.
            auto seq = dorado::utils::extract_sequence(rec);
            CHECK(nonbc_seq == seq);

            auto qual = dorado::utils::extract_quality(rec);
            CHECK(qual.size() == seq.length());

            CHECK(bam_message.primer_classification.primer_name == "cDNA_FWD");
            CHECK(bam_message.primer_classification.orientation == 1);
            CHECK(bam_aux2A(bam_aux_get(rec, "TS")) == '+');

            auto [_, move_vals] = dorado::utils::extract_move_table(rec);
            CHECK(move_vals == expected_move_vals);

            // The mod should now be at the very first base.
            const std::string expected_mod_str = "A+a.,0;";
            const std::vector<uint8_t> expected_mod_probs = {235};
            auto [mod_str, mod_probs] = dorado::utils::extract_modbase_info(rec);
            CHECK(mod_str == expected_mod_str);
            CHECK_THAT(mod_probs, Equals(std::vector<uint8_t>{235}));

            CHECK(bam_aux2i(bam_aux_get(rec, "ts")) == additional_trimmed_samples);
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            // Check trimming on the Read type.
            auto msg_read = std::get<SimplexReadPtr>(std::move(message));
            const ReadCommon& read_common = msg_read->read_common;

            CHECK(read_common.seq == nonbc_seq);

            CHECK(read_common.moves == expected_move_vals);

            CHECK(read_common.primer_classification.primer_name == "cDNA_FWD");
            CHECK(read_common.primer_classification.orientation == 1);

            // The mod probabilities table should now start mod at the first base.
            CHECK(read_common.base_mod_probs.size() ==
                  read_common.seq.size() * mod_alphabet.size());
            CHECK(read_common.base_mod_probs[0] == 20);
            CHECK(read_common.base_mod_probs[1] == 235);

            CHECK(read_common.num_trimmed_samples == uint64_t(additional_trimmed_samples));

            auto bams = read_common.extract_sam_lines(0, 10, false);
            auto& rec = bams[0];
            auto [mod_str, mod_probs] = dorado::utils::extract_modbase_info(rec.get());
        }
    }
}
