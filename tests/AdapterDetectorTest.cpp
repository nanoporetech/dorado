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
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <htslib/sam.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#define TEST_GROUP "[adapter_detect]"

namespace {
const std::string TEST_KIT1 = "SQK-LSK114";
const std::string TEST_KIT2 = "SQK-PCS114";

using Query = dorado::demux::AdapterDetector::Query;

void sort_queries(std::vector<Query>& queries) {
    auto query_lt = [](const Query& a, const Query& b) -> bool { return a.name < b.name; };
    std::sort(queries.begin(), queries.end(), query_lt);
}

}  // namespace

namespace fs = std::filesystem;

using namespace dorado;

CATCH_TEST_CASE("AdapterDetector: test classification", TEST_GROUP) {
    demux::AdapterDetector detector(std::nullopt);
    AdapterScoreResult res;
    std::string dummy_sequence{};
    std::pair<int, int> dummy_interval{};

    // Consistent 5' primer at front and 3' primer at rear
    res.front.name = "test_FWD_FRONT";
    res.rear.name = "test_FWD_REAR";
    auto classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == "test_FWD");
    CATCH_CHECK(classification.orientation == StrandOrientation::FORWARD);
    CATCH_CHECK(to_char(classification.orientation) == '+');

    // Only 5' primer at front
    res.front.name = "test_FWD_FRONT";
    res.rear.name = UNCLASSIFIED;
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == "test_FWD");
    CATCH_CHECK(classification.orientation == StrandOrientation::FORWARD);
    CATCH_CHECK(to_char(classification.orientation) == '+');

    // Only 3' primer at rear
    res.front.name = UNCLASSIFIED;
    res.rear.name = "test_FWD_REAR";
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == "test_FWD");
    CATCH_CHECK(classification.orientation == StrandOrientation::FORWARD);
    CATCH_CHECK(to_char(classification.orientation) == '+');

    // Consistent 3' primer at front and 5' primer at rear
    res.front.name = "test_REV_FRONT";
    res.rear.name = "test_REV_REAR";
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == "test_REV");
    CATCH_CHECK(classification.orientation == StrandOrientation::REVERSE);
    CATCH_CHECK(to_char(classification.orientation) == '-');

    // Only 3' primer at front
    res.front.name = "test_REV_FRONT";
    res.rear.name = UNCLASSIFIED;
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == "test_REV");
    CATCH_CHECK(classification.orientation == StrandOrientation::REVERSE);
    CATCH_CHECK(to_char(classification.orientation) == '-');

    // Only 5' primer at rear
    res.front.name = UNCLASSIFIED;
    res.rear.name = "test_REV_REAR";
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == "test_REV");
    CATCH_CHECK(classification.orientation == StrandOrientation::REVERSE);
    CATCH_CHECK(to_char(classification.orientation) == '-');

    // No primers found at either end
    res.front.name = UNCLASSIFIED;
    res.rear.name = UNCLASSIFIED;
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == UNCLASSIFIED);
    CATCH_CHECK(classification.orientation == StrandOrientation::UNKNOWN);
    CATCH_CHECK(to_char(classification.orientation) == '?');

    // Inconsistent 5' primer at front and 3' primer at rear
    res.front.name = "test1_FWD_FRONT";
    res.rear.name = "test2_FWD_REAR";
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == UNCLASSIFIED);
    CATCH_CHECK(classification.orientation == StrandOrientation::UNKNOWN);
    CATCH_CHECK(to_char(classification.orientation) == '?');

    // 5' primer found at both front and rear
    res.front.name = "test1_FWD_FRONT";
    res.rear.name = "test1_REV_REAR";
    classification = detector.classify_primers(res, dummy_interval, dummy_sequence);
    CATCH_CHECK(classification.primer_name == UNCLASSIFIED);
    CATCH_CHECK(classification.orientation == StrandOrientation::UNKNOWN);
    CATCH_CHECK(to_char(classification.orientation) == '?');
}

CATCH_TEST_CASE("AdapterDetector: test UMI detection", TEST_GROUP) {
    // Create new read that is [PCS110_SSP_FWD] - [UMI_TAG] - 200 As - [PCS110_VNP_REV].
    demux::AdapterDetector detector(std::nullopt);
    const std::string nonbc_seq = std::string(200, 'A');
    const auto& primers = detector.get_primer_sequences(TEST_KIT2);
    const auto& front_primer = primers[0].front_sequence;
    const auto& rear_primer = primers[0].rear_sequence;
    const std::string umi_partial = "AAAATTCCCCTTGGGGTTACGATTT";
    const auto sequence = front_primer + umi_partial + nonbc_seq + rear_primer;
    const std::string umi_full = "TTT" + umi_partial;
    auto trim_start = int(front_primer.size());
    auto trim_end = int(front_primer.size() + umi_partial.size()) + 200;
    auto trim_interval = std::make_pair(trim_start, trim_end);
    AdapterScoreResult res;
    res.front.name = "PCS110_FWD_FRONT";
    res.front.position = {0, trim_start - 1};
    res.front.score = 1.0f;
    res.rear.name = "PCS110_FWD_REAR";
    res.rear.position = {trim_end, int(sequence.size() - 1)};
    res.rear.score = 1.0f;
    auto classification = detector.classify_primers(res, trim_interval, sequence);
    CATCH_CHECK(to_char(classification.orientation) == '+');
    CATCH_CHECK(trim_interval.first == trim_start + int(umi_partial.size()));
    CATCH_CHECK(trim_interval.second == trim_end);
    CATCH_CHECK(classification.umi_tag_sequence == umi_full);

    // Do the same for a reverse strand.
    auto rev_sequence = utils::reverse_complement(sequence);
    trim_start = int(rear_primer.size());
    trim_end = int(rear_primer.size() + umi_partial.size()) + 200;
    trim_interval = std::make_pair(trim_start, trim_end);
    res.front.name = "PCS110_REV_FRONT";
    res.front.position = {0, trim_start - 1};
    res.rear.name = "PCS110_REV_REAR";
    res.rear.position = {trim_end, int(rev_sequence.size() - 1)};
    classification = detector.classify_primers(res, trim_interval, rev_sequence);
    CATCH_CHECK(to_char(classification.orientation) == '-');
    CATCH_CHECK(trim_interval.first == trim_start);
    CATCH_CHECK(trim_interval.second == trim_end - int(umi_partial.size()));
    CATCH_CHECK(classification.umi_tag_sequence == umi_full);
}

CATCH_TEST_CASE("AdapterDetector: test adapter detection", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::AdapterDetector detector(std::nullopt);
    const auto& adapters = detector.get_adapter_sequences(TEST_KIT1);

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < adapters.size(); ++i) {
        // First put the front adapter only at the beginning, with 6 bases in front of it.
        const auto& adapter = adapters[i];
        auto new_sequence1 = "ACGTAC" + adapter.front_sequence + seq;
        auto res1 = detector.find_adapters(new_sequence1, TEST_KIT1);
        CATCH_CHECK(res1.front.name == adapter.name + "_FRONT");
        CATCH_CHECK(res1.front.position ==
                    std::make_pair(6, int(adapter.front_sequence.length()) + 5));
        CATCH_CHECK(res1.front.score == 1.0f);
        CATCH_CHECK(res1.rear.score < 0.7f);

        // Now put the rear adapter at the end, with 3 bases after it.
        auto new_sequence2 = seq + adapter.rear_sequence + "TTT";
        auto res2 = detector.find_adapters(new_sequence2, TEST_KIT1);
        CATCH_CHECK(res2.front.score < 0.7f);
        CATCH_CHECK(res2.rear.name == adapter.name + "_REAR");
        CATCH_CHECK(res2.rear.position ==
                    std::make_pair(int(seq.length()),
                                   int(seq.length() + adapter.rear_sequence.length()) - 1));
        CATCH_CHECK(res2.rear.score == 1.0f);

        // Now put them both in.
        auto new_sequence3 = "TGCA" + adapter.front_sequence + seq + adapter.rear_sequence + "GTA";
        auto res3 = detector.find_adapters(new_sequence3, TEST_KIT1);
        CATCH_CHECK(res3.front.name == adapter.name + "_FRONT");
        CATCH_CHECK(res3.front.position ==
                    std::make_pair(4, int(adapter.front_sequence.length()) + 3));
        CATCH_CHECK(res3.front.score == 1.0f);
        CATCH_CHECK(res3.rear.name == adapter.name + "_REAR");
        CATCH_CHECK(res3.rear.position ==
                    std::make_pair(int(adapter.front_sequence.length() + seq.length()) + 4,
                                   int(adapter.front_sequence.length() + seq.length() +
                                       adapter.rear_sequence.length()) +
                                           3));
        CATCH_CHECK(res3.rear.score == 1.0f);
    }
}

CATCH_TEST_CASE("AdapterDetector: test primer detection", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::AdapterDetector detector(std::nullopt);
    auto primers = detector.get_primer_sequences(TEST_KIT1);
    sort_queries(primers);

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    CATCH_CHECK(primers.size() == 2);
    for (size_t i = 0; i < primers.size(); ++i) {
        const auto& primer = primers[i];
        auto new_sequence1 = "ACGTAC" + primer.front_sequence + seq + primer.rear_sequence + "TTT";
        auto res = detector.find_primers(new_sequence1, TEST_KIT1);
        CATCH_CHECK(res.front.name == primer.name + "_FRONT");
        CATCH_CHECK(res.front.position ==
                    std::make_pair(6, int(primer.front_sequence.length()) + 5));
        CATCH_CHECK(res.front.score == 1.0f);
        CATCH_CHECK(res.rear.name == primer.name + "_REAR");
        CATCH_CHECK(res.rear.position ==
                    std::make_pair(int(primer.front_sequence.length() + seq.length()) + 6,
                                   int(primer.front_sequence.length() + seq.length() +
                                       primer.rear_sequence.length()) +
                                           5));
        CATCH_CHECK(res.rear.score == 1.0f);
        std::pair<int, int> dummy_interval{};
        auto classification = detector.classify_primers(res, dummy_interval, seq);
        CATCH_CHECK(classification.primer_name == primer.name);
        StrandOrientation expected_orientation =
                (i == 0) ? StrandOrientation::FORWARD : StrandOrientation::REVERSE;
        CATCH_CHECK(classification.orientation == expected_orientation);
    }
}

CATCH_TEST_CASE("AdapterDetector: test custom primer detection with kit", TEST_GROUP) {
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
    CATCH_REQUIRE(adapters.size() == 1);
    CATCH_CHECK(adapters[0].name == expected_adapter_name);
    CATCH_CHECK(adapters[0].front_sequence == expected_adapter_front);
    CATCH_CHECK(adapters[0].rear_sequence == expected_adapter_rear);

    CATCH_REQUIRE(primers.size() == 2);
    CATCH_CHECK(primers[0].name == expected_primer_names[0]);
    CATCH_CHECK(primers[0].front_sequence == expected_primer_fronts[0]);
    CATCH_CHECK(primers[0].rear_sequence == expected_primer_rears[0]);
    CATCH_CHECK(primers[1].name == expected_primer_names[1]);
    CATCH_CHECK(primers[1].front_sequence == expected_primer_fronts[1]);
    CATCH_CHECK(primers[1].rear_sequence == expected_primer_rears[1]);

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < primers.size(); ++i) {
        // Put the front primer at the beginning, and the rear primer at the end.
        const auto& primer = primers[i];
        auto new_sequence1 = "ACGTAC" + primer.front_sequence + seq + primer.rear_sequence + "TTT";
        auto res = detector.find_primers(new_sequence1, "TEST_KIT2");
        CATCH_CHECK(res.front.name == primer.name + "_FRONT");
        CATCH_CHECK(res.front.position ==
                    std::make_pair(6, int(primer.front_sequence.length()) + 5));
        CATCH_CHECK(res.front.score == 1.0f);
        CATCH_CHECK(res.rear.name == primer.name + "_REAR");
        CATCH_CHECK(res.rear.position ==
                    std::make_pair(int(primer.front_sequence.length() + seq.length()) + 6,
                                   int(primer.front_sequence.length() + seq.length() +
                                       primer.rear_sequence.length()) +
                                           5));
        CATCH_CHECK(res.rear.score == 1.0f);
        std::pair<int, int> dummy_interval{};
        auto classification = detector.classify_primers(res, dummy_interval, seq);
        CATCH_CHECK(classification.primer_name == primers[i].name);
        StrandOrientation expected_orientation =
                (i == 0) ? StrandOrientation::FORWARD : StrandOrientation::REVERSE;
        CATCH_CHECK(classification.orientation == expected_orientation);
    }
}

CATCH_TEST_CASE("AdapterDetector: test custom primer detection without kit", TEST_GROUP) {
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
    CATCH_CHECK(adapters.size() == 1);
    CATCH_CHECK(primers.size() == 4);
    for (size_t i = 0; i < primers.size(); ++i) {
        CATCH_CHECK(primers[i].name == expected_primer_names[i]);
        CATCH_CHECK(primers[i].front_sequence == expected_primers_front[i]);
        CATCH_CHECK(primers[i].rear_sequence == expected_primers_rear[i]);
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
        CATCH_CHECK(res.front.name == primers[i].name + "_FRONT");
        CATCH_CHECK(res.front.position ==
                    std::make_pair(6, int(primers[i].front_sequence.length()) + 5));
        CATCH_CHECK(res.front.score == 1.0f);
        CATCH_CHECK(res.rear.name == primers[i].name + "_REAR");
        CATCH_CHECK(res.rear.position ==
                    std::make_pair(int(primers[i].front_sequence.length() + seq.length()) + 6,
                                   int(primers[i].front_sequence.length() + seq.length() +
                                       primers[i].rear_sequence.length()) +
                                           5));
        CATCH_CHECK(res.rear.score == 1.0f);
        std::pair<int, int> dummy_interval{};
        auto classification = detector.classify_primers(res, dummy_interval, seq);
        CATCH_CHECK(classification.primer_name == primers[i].name);
        StrandOrientation expected_orientation =
                (i % 2 == 0) ? StrandOrientation::FORWARD : StrandOrientation::REVERSE;
        CATCH_CHECK(classification.orientation == expected_orientation);
    }
}

CATCH_TEST_CASE(
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

    // Create new read that is [LSK110_FWD] - [PCS110_SSP_FWD] - [UMI_TAG] - 200 As - [PCS110_VNP_REV] [LSK110_REV].
    const std::string umi_partial = "AAAATTCCCCTTGGGGTTACGATTT";
    const std::string umi_full = "TTT" + umi_partial;
    auto read = std::make_unique<SimplexRead>();
    const std::string nonbc_seq = std::string(200, 'A');
    demux::AdapterDetector detector(std::nullopt);
    const auto& adapters = detector.get_adapter_sequences(TEST_KIT2);
    const auto& primers = detector.get_primer_sequences(TEST_KIT2);
    const auto& front_adapter = adapters[0].front_sequence;
    const auto& front_primer = primers[0].front_sequence;
    const auto& rear_adapter = adapters[0].rear_sequence;
    const auto& rear_primer = primers[0].rear_sequence;
    const int stride = 6;
    read->read_common.seq =
            front_adapter + front_primer + umi_partial + nonbc_seq + rear_primer + rear_adapter;
    read->read_common.qstring = std::string(read->read_common.seq.length(), '!');
    read->read_common.read_id = "read_id";
    read->read_common.model_stride = stride;
    read->read_common.sequencing_kit = TEST_KIT2;

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
    auto flank_size = front_adapter.length() + front_primer.length() + umi_partial.length();
    read->read_common.base_mod_probs[flank_size * mod_alphabet.size()] = 20;         // A
    read->read_common.base_mod_probs[(flank_size * mod_alphabet.size()) + 1] = 235;  // 6mA
    read->read_common.num_trimmed_samples = 0;

    auto records = read->read_common.extract_sam_lines(true /* emit moves */,
                                                       static_cast<uint8_t>(10), false);
    BamPtr record_copy(bam_dup1(records[0].get()));

    auto client_info = std::make_shared<dorado::DefaultClientInfo>();
    client_info->contexts().register_context<const dorado::demux::AdapterInfo>(
            std::make_shared<const dorado::demux::AdapterInfo>(
                    dorado::demux::AdapterInfo{true, true, false, std::nullopt}));
    read->read_common.client_info = std::move(client_info);

    BamMessage bam_read{std::move(record_copy), read->read_common.client_info};
    bam_read.sequencing_kit = TEST_KIT2;

    // Push a Symplex read type.
    pipeline->push_message(std::move(read));

    // Push a BAM read type.
    pipeline->push_message(std::move(bam_read));

    // Push a type not used by the ClassifierNode.
    dorado::ReadPair dummy_read_pair;
    pipeline->push_message(std::move(dummy_read_pair));

    pipeline->terminate(DefaultFlushOptions());

    const size_t num_expected_messages = 3;
    CATCH_CHECK(messages.size() == num_expected_messages);

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
            CATCH_CHECK(nonbc_seq == seq);

            auto qual = dorado::utils::extract_quality(rec);
            CATCH_CHECK(qual.size() == seq.length());

            CATCH_CHECK(bam_message.primer_classification.primer_name == "PCS110_FWD");
            CATCH_CHECK(bam_message.primer_classification.orientation ==
                        StrandOrientation::FORWARD);
            CATCH_CHECK(bam_aux2A(bam_aux_get(rec, "TS")) == '+');
            std::string expected_umi_tag = "RX:Z:" + umi_full;
            kstring_t umi_buffer = KS_INITIALIZE;
            CATCH_CHECK(bam_aux_get_str(rec, "RX", &umi_buffer) == 1);
            CATCH_CHECK(umi_buffer.l == expected_umi_tag.size());
            auto umi_tag = (umi_buffer.s != nullptr) ? std::string(umi_buffer.s, umi_buffer.l)
                                                     : std::string{};
            CATCH_CHECK(umi_tag == expected_umi_tag);
            ks_free(&umi_buffer);

            auto [_, move_vals] = dorado::utils::extract_move_table(rec);
            CATCH_CHECK(move_vals == expected_move_vals);

            // The mod should now be at the very first base.
            const std::string expected_mod_str = "A+a.,0;";
            const std::vector<uint8_t> expected_mod_probs = {235};
            auto [mod_str, mod_probs] = dorado::utils::extract_modbase_info(rec);
            CATCH_CHECK(mod_str == expected_mod_str);
            CATCH_CHECK_THAT(mod_probs, Equals(std::vector<uint8_t>{235}));

            CATCH_CHECK(bam_aux2i(bam_aux_get(rec, "ts")) == additional_trimmed_samples);
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            // Check trimming on the Read type.
            auto msg_read = std::get<SimplexReadPtr>(std::move(message));
            const ReadCommon& read_common = msg_read->read_common;

            CATCH_CHECK(read_common.seq == nonbc_seq);

            CATCH_CHECK(read_common.moves == expected_move_vals);

            CATCH_CHECK(read_common.primer_classification.primer_name == "PCS110_FWD");
            CATCH_CHECK(read_common.primer_classification.orientation ==
                        StrandOrientation::FORWARD);
            CATCH_CHECK(read_common.primer_classification.umi_tag_sequence == umi_full);

            // The mod probabilities table should now start mod at the first base.
            CATCH_CHECK(read_common.base_mod_probs.size() ==
                        read_common.seq.size() * mod_alphabet.size());
            CATCH_CHECK(read_common.base_mod_probs[0] == 20);
            CATCH_CHECK(read_common.base_mod_probs[1] == 235);

            CATCH_CHECK(read_common.num_trimmed_samples == uint64_t(additional_trimmed_samples));

            auto bams = read_common.extract_sam_lines(0, static_cast<uint8_t>(10), false);
            auto& rec = bams[0];
            auto [mod_str, mod_probs] = dorado::utils::extract_modbase_info(rec.get());
        }
    }
}
