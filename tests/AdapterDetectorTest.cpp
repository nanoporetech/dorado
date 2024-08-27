#include "demux/AdapterDetector.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "demux/Trimmer.h"
#include "demux/adapter_info.h"
#include "read_pipeline/AdapterDetectorNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
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

namespace fs = std::filesystem;

using namespace dorado;

TEST_CASE("AdapterDetector: test adapter detection", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::AdapterDetector detector(std::nullopt);
    const auto& adapters = detector.get_adapter_sequences();

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < adapters.size(); ++i) {
        // First put the front adapter only at the beginning, with 6 bases in front of it.
        if (!adapters[i].sequence.empty()) {
            auto new_sequence1 = "ACGTAC" + adapters[i].sequence + seq;
            auto res1 = detector.find_adapters(new_sequence1);
            CHECK(res1.front.name == adapters[i].name + "_FWD");
            CHECK(res1.front.position == std::make_pair(6, int(adapters[i].sequence.length()) + 5));
            CHECK(res1.front.score == 1.0f);
            CHECK(res1.rear.score < 0.7f);
        }

        // Now put the rear adapter at the end, with 3 bases after it.
        if (!adapters[i].sequence_rev.empty()) {
            auto new_sequence2 = seq + adapters[i].sequence_rev + "TTT";
            auto res2 = detector.find_adapters(new_sequence2);
            CHECK(res2.front.score < 0.7f);
            CHECK(res2.rear.name == adapters[i].name + "_REV");
            CHECK(res2.rear.position ==
                  std::make_pair(int(seq.size()),
                                 int(seq.size() + adapters[i].sequence_rev.length()) - 1));
            CHECK(res2.rear.score == 1.0f);
        }

        // Now put them both in.
        if (!adapters[i].sequence.empty() && !adapters[i].sequence_rev.empty()) {
            auto new_sequence3 =
                    "TGCA" + adapters[i].sequence + seq + adapters[i].sequence_rev + "GTA";
            auto res3 = detector.find_adapters(new_sequence3);
            CHECK(res3.front.name == adapters[i].name + "_FWD");
            CHECK(res3.front.position == std::make_pair(4, int(adapters[i].sequence.length()) + 3));
            CHECK(res3.front.score == 1.0f);
            CHECK(res3.rear.name == adapters[i].name + "_REV");
            CHECK(res3.rear.position ==
                  std::make_pair(int(adapters[i].sequence.size() + seq.size()) + 4,
                                 int(adapters[i].sequence.size() + seq.size() +
                                     adapters[i].sequence_rev.length()) +
                                         3));
            CHECK(res3.rear.score == 1.0f);
        }
    }
}

TEST_CASE("AdapterDetector: test primer detection", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));

    demux::AdapterDetector detector(std::nullopt);
    const auto& primers = detector.get_primer_sequences();

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < primers.size(); ++i) {
        // First put the primer at the beginning, and its reverse at the end.
        auto new_sequence1 = "ACGTAC" + primers[i].sequence + seq + primers[i].sequence_rev + "TTT";
        auto res1 = detector.find_primers(new_sequence1);
        CHECK(res1.front.name == primers[i].name + "_FWD");
        CHECK(res1.front.position == std::make_pair(6, int(primers[i].sequence.length()) + 5));
        CHECK(res1.front.score == 1.0f);
        CHECK(res1.rear.name == primers[i].name + "_REV");
        CHECK(res1.rear.position ==
              std::make_pair(int(primers[i].sequence.length() + seq.length()) + 6,
                             int(primers[i].sequence.length() + seq.length() +
                                 primers[i].sequence_rev.length()) +
                                     5));
        CHECK(res1.rear.score == 1.0f);

        // Now put the reverse primers at the beginning, and the forward ones at the end.
        auto new_sequence2 = "ACGTAC" + primers[i].sequence_rev + seq + primers[i].sequence + "TTT";
        auto res2 = detector.find_primers(new_sequence2);
        CHECK(res2.front.name == primers[i].name + "_REV");
        CHECK(res2.front.position == std::make_pair(6, int(primers[i].sequence_rev.length()) + 5));
        CHECK(res2.front.score == 1.0f);
        CHECK(res2.rear.name == primers[i].name + "_FWD");
        CHECK(res2.rear.position ==
              std::make_pair(int(primers[i].sequence_rev.length() + seq.length()) + 6,
                             int(primers[i].sequence_rev.length() + seq.length() +
                                 primers[i].sequence.length()) +
                                     5));
        CHECK(res2.rear.score == 1.0f);
    }
}

TEST_CASE("AdapterDetector: test custom primer detection", TEST_GROUP) {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/single_end"));
    fs::path seq_dir = fs::path(get_data_dir("adapter_trim"));
    auto custom_primer_file = (seq_dir / "custom_primers.fasta").string();

    demux::AdapterDetector detector(custom_primer_file);
    const auto& primers = detector.get_primer_sequences();
    // Make sure the primers have been properly loaded.
    std::vector<std::string> expected_names = {"primer1", "primer2"};
    std::vector<std::string> expected_seqs = {"TGCGAAT", "GACCTCTG"};
    std::vector<std::string> expected_rc_seqs = {"ATTCGCA", "CAGAGGTC"};
    CHECK(primers.size() == expected_names.size());
    for (size_t i = 0; i < primers.size(); ++i) {
        CHECK(primers[i].name == expected_names[i]);
        CHECK(primers[i].sequence == expected_seqs[i]);
        CHECK(primers[i].sequence_rev == expected_rc_seqs[i]);
    }

    auto test_file = data_dir / "SQK-RBK114-96_BC01.fastq";
    HtsReader reader(test_file.string(), std::nullopt);
    reader.read();
    std::string seq = utils::extract_sequence(reader.record.get());
    for (size_t i = 0; i < primers.size(); ++i) {
        // First put the primer at the beginning, and its reverse at the end.
        auto new_sequence1 = "ACGTAC" + primers[i].sequence + seq + primers[i].sequence_rev + "TTT";
        auto res1 = detector.find_primers(new_sequence1);
        CHECK(res1.front.name == primers[i].name + "_FWD");
        CHECK(res1.front.position == std::make_pair(6, int(primers[i].sequence.length()) + 5));
        CHECK(res1.front.score == 1.0f);
        CHECK(res1.rear.name == primers[i].name + "_REV");
        CHECK(res1.rear.position ==
              std::make_pair(int(primers[i].sequence.length() + seq.length()) + 6,
                             int(primers[i].sequence.length() + seq.length() +
                                 primers[i].sequence_rev.length()) +
                                     5));
        CHECK(res1.rear.score == 1.0f);

        // Now put the reverse primers at the beginning, and the forward ones at the end.
        auto new_sequence2 = "ACGTAC" + primers[i].sequence_rev + seq + primers[i].sequence + "TTT";
        auto res2 = detector.find_primers(new_sequence2);
        CHECK(res2.front.name == primers[i].name + "_REV");
        CHECK(res2.front.position == std::make_pair(6, int(primers[i].sequence_rev.length()) + 5));
        CHECK(res2.front.score == 1.0f);
        CHECK(res2.rear.name == primers[i].name + "_FWD");
        CHECK(res2.rear.position ==
              std::make_pair(int(primers[i].sequence_rev.length() + seq.length()) + 6,
                             int(primers[i].sequence_rev.length() + seq.length() +
                                 primers[i].sequence.length()) +
                                     5));
        CHECK(res2.rear.score == 1.0f);
    }
}

void detect_and_trim(SimplexRead& read) {
    demux::AdapterDetector detector(std::nullopt);
    auto seqlen = int(read.read_common.seq.length());
    std::pair<int, int> adapter_trim_interval = {0, seqlen};
    std::pair<int, int> primer_trim_interval = {0, seqlen};

    auto adapter_res = detector.find_adapters(read.read_common.seq);
    adapter_trim_interval = Trimmer::determine_trim_interval(adapter_res, seqlen);
    auto primer_res = detector.find_primers(read.read_common.seq);
    primer_trim_interval = Trimmer::determine_trim_interval(primer_res, seqlen);
    std::pair<int, int> trim_interval = adapter_trim_interval;
    trim_interval.first = std::max(trim_interval.first, primer_trim_interval.first);
    trim_interval.second = std::min(trim_interval.second, primer_trim_interval.second);
    CHECK(trim_interval.first < trim_interval.second);
    demux::AdapterDetector::check_and_update_barcoding(read, trim_interval);
    Trimmer::trim_sequence(read, trim_interval);
    read.read_common.adapter_trim_interval = trim_interval;
}

TEST_CASE(
        "AdapterDetector: check trimming when adapter/primer trimming is combined with barcode "
        "detection.",
        TEST_GROUP) {
    using Catch::Matchers::Equals;

    const std::string seq = std::string(200, 'A');
    demux::AdapterDetector detector(std::nullopt);
    const auto& adapters = detector.get_adapter_sequences();
    const auto& primers = detector.get_primer_sequences();
    const auto& front_adapter = adapters[1].sequence;
    const auto& front_primer = primers[2].sequence;
    const auto& rear_adapter = adapters[1].sequence_rev;
    const auto& rear_primer = primers[2].sequence_rev;
    std::string front_barcode = "CCCCCCCCCC";
    std::string rear_barcode = "GGGGGGGGGG";
    const int stride = 6;

    {
        // Test case where barcode detection has been done, but barcodes were not trimmed.
        // Make sure that barcode results are updated to reflect their position in the sequence
        // after the front adapter and primer have been trimmed.
        SimplexRead read;
        read.read_common.seq = front_adapter + front_primer + front_barcode + seq + rear_barcode +
                               rear_primer + rear_adapter;
        read.read_common.qstring = std::string(read.read_common.seq.length(), '!');
        read.read_common.read_id = "read_id";
        read.read_common.model_stride = stride;
        read.read_common.num_trimmed_samples = 0;
        read.read_common.pre_trim_seq_length = read.read_common.seq.length();

        std::vector<uint8_t> moves;
        for (size_t i = 0; i < read.read_common.seq.length(); i++) {
            moves.push_back(1);
            moves.push_back(0);
        }
        read.read_common.moves = moves;
        read.read_common.raw_data = at::zeros(moves.size() * stride);

        const auto flank_size = front_adapter.length() + front_primer.length();
        const int additional_trimmed_samples =
                int(stride * 2 * flank_size);  // * 2 is because we have 2 moves per base

        // Add in barcoding information.
        read.read_common.barcoding_result = std::make_shared<BarcodeScoreResult>();
        auto& barcode_results = *read.read_common.barcoding_result;
        barcode_results.barcode_name = "fake_barcode";
        int front_barcode_start = int(front_adapter.length() + front_primer.length());
        int front_barcode_end = front_barcode_start + int(front_barcode.length());
        int rear_barcode_start = front_barcode_end + int(seq.length());
        int rear_barcode_end = rear_barcode_start + int(rear_barcode.length());
        barcode_results.top_barcode_pos = {front_barcode_start, front_barcode_end};
        barcode_results.bottom_barcode_pos = {rear_barcode_start, rear_barcode_end};

        detect_and_trim(read);
        std::string expected_trimmed_seq = front_barcode + seq + rear_barcode;
        CHECK(read.read_common.seq == expected_trimmed_seq);
        CHECK(read.read_common.num_trimmed_samples == uint64_t(additional_trimmed_samples));
        int expected_front_barcode_start = 0;
        int expected_front_barcode_end = int(front_barcode.length());
        int expected_rear_barcode_start = expected_front_barcode_end + int(seq.length());
        int expected_rear_barcode_end = expected_rear_barcode_start + int(rear_barcode.length());
        CHECK(barcode_results.top_barcode_pos ==
              std::pair<int, int>(expected_front_barcode_start, expected_front_barcode_end));
        CHECK(barcode_results.bottom_barcode_pos ==
              std::pair<int, int>(expected_rear_barcode_start, expected_rear_barcode_end));
    }
    {
        // Test case where barcode detection has been done, but barcodes were not trimmed.
        // In this case the detected adapter/primer overlaps what was detected as the barcode.
        // The code should therefore not trim anything.
        SimplexRead read;
        read.read_common.seq = front_adapter + front_primer + seq + rear_primer + rear_adapter;
        read.read_common.qstring = std::string(read.read_common.seq.length(), '!');
        read.read_common.read_id = "read_id";
        read.read_common.model_stride = stride;
        read.read_common.num_trimmed_samples = 0;
        read.read_common.pre_trim_seq_length = read.read_common.seq.length();

        std::vector<uint8_t> moves;
        for (size_t i = 0; i < read.read_common.seq.length(); i++) {
            moves.push_back(1);
            moves.push_back(0);
        }
        read.read_common.moves = moves;
        read.read_common.raw_data = at::zeros(moves.size() * stride);

        // Add in barcoding information.
        read.read_common.barcoding_result = std::make_shared<BarcodeScoreResult>();
        auto& barcode_results = *read.read_common.barcoding_result;
        barcode_results.barcode_name = "fake_barcode";
        int front_barcode_start = 5;
        int front_barcode_end = 15;
        int rear_barcode_start =
                int(front_adapter.length() + front_primer.length() + seq.length()) + 5;
        int rear_barcode_end = rear_barcode_start + 10;
        barcode_results.top_barcode_pos = {front_barcode_start, front_barcode_end};
        barcode_results.bottom_barcode_pos = {rear_barcode_start, rear_barcode_end};

        std::string expected_trimmed_seq = read.read_common.seq;  // Nothing should get trimmed.
        detect_and_trim(read);
        CHECK(read.read_common.seq == expected_trimmed_seq);
        CHECK(read.read_common.num_trimmed_samples == uint64_t(0));
        CHECK(barcode_results.top_barcode_pos ==
              std::pair<int, int>(front_barcode_start, front_barcode_end));
        CHECK(barcode_results.bottom_barcode_pos ==
              std::pair<int, int>(rear_barcode_start, rear_barcode_end));
    }
}

TEST_CASE(
        "AdapterDetectorNode: check read messages are correctly updated after adapter/primer "
        "detection and trimming",
        TEST_GROUP) {
    using Catch::Matchers::Equals;

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    pipeline_desc.add_node<AdapterDetectorNode>({sink}, 8);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    // Create new read that is [LSK110_FWD] - [cDNA_VNP_FWD] - 200 As - [cDNA_VNP_REV] [LSK110_REV].
    auto read = std::make_unique<SimplexRead>();
    const std::string nonbc_seq = std::string(200, 'A');
    demux::AdapterDetector detector(std::nullopt);
    const auto& adapters = detector.get_adapter_sequences();
    const auto& primers = detector.get_primer_sequences();
    const auto& front_adapter = adapters[1].sequence;
    const auto& front_primer = primers[2].sequence;
    const auto& rear_adapter = adapters[1].sequence_rev;
    const auto& rear_primer = primers[2].sequence_rev;
    const int stride = 6;
    read->read_common.seq = front_adapter + front_primer + nonbc_seq + rear_primer + rear_adapter;
    read->read_common.qstring = std::string(read->read_common.seq.length(), '!');
    read->read_common.read_id = "read_id";
    read->read_common.model_stride = stride;

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

    auto client_info = std::make_shared<dorado::DefaultClientInfo>();
    client_info->contexts().register_context<const dorado::demux::AdapterInfo>(
            std::make_shared<const dorado::demux::AdapterInfo>(
                    dorado::demux::AdapterInfo{true, true, std::nullopt}));
    read->read_common.client_info = std::move(client_info);

    // Push a Read type.
    pipeline->push_message(std::move(read));

    // Push a type not used by the ClassifierNode.
    dorado::ReadPair dummy_read_pair;
    pipeline->push_message(std::move(dummy_read_pair));

    pipeline->terminate(DefaultFlushOptions());

    const size_t num_expected_messages = 2;
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
