#include "read_pipeline/PairingNode.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "utils/fasta_reader.h"
#include "utils/sequence_utils.h"
#include "utils/time_utils.h"

#include <ATen/Functions.h>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#define TEST_GROUP "[PairingNodeTest]"

namespace {

// Generate a read with a specified start time delay.
auto make_read(int delay_ms, std::string seq) {
    auto read = std::make_unique<dorado::SimplexRead>();
    read->read_common.raw_data = at::zeros({10});
    read->read_common.sample_rate = 4000;
    read->read_common.num_trimmed_samples = 10;
    read->read_common.attributes.channel_number = 664;
    read->read_common.attributes.mux = 3;
    read->read_common.attributes.num_samples = 10000;
    read->start_sample = 29767426 + (delay_ms * read->read_common.sample_rate) / 1000;
    read->end_sample = read->start_sample + read->read_common.attributes.num_samples;
    read->run_acquisition_start_time_ms = 1676976119670;
    read->read_common.start_time_ms =
            read->run_acquisition_start_time_ms +
            uint64_t(std::round(read->start_sample * 1000. / read->read_common.sample_rate));
    read->read_common.attributes.start_time =
            dorado::utils::get_string_timestamp_from_unix_time(read->read_common.start_time_ms);
    read->read_common.qstring = std::string(seq.length(), '~');
    read->read_common.seq = std::move(seq);
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();

    return read;
}

// Generate a read that is seq_len long with a specified start
// time delay.
auto make_read(int delay_ms, size_t seq_len) {
    return make_read(delay_ms, std::string(seq_len, 'A'));
}

}  // namespace

CATCH_TEST_CASE("Split read pairing", TEST_GROUP) {
    // the second read must start within 1000ms of the end of the first read
    // and min/max length ratio must be greater than 0.2
    // expected pairs: {2, 3} and {5, 6}

    // Load a pre-determined read to exercise the mapping pathway.
    const auto fa_file = std::filesystem::path(get_aligner_data_dir()) / "long_target.fa";
    dorado::utils::FastaReader fa_reader(fa_file.string());
    auto record = fa_reader.try_get_next_record();
    record = fa_reader.try_get_next_record();  // Skip the first sequence and use the second one.
    CATCH_REQUIRE(record.has_value());
    const std::string seq = record->sequence();
    auto seq_rc = dorado::utils::reverse_complement(seq);
    seq_rc = seq_rc.substr(0, size_t(seq.length() * 0.8f));

    std::array reads{
            make_read(0, 1000),       //
            make_read(10, 1000),      // too early to pair with {0}
            make_read(10000, 6000),   // too late to pair with {1}
            make_read(12500, 5990),   //
            make_read(18000, 100),    // too short to pair with {2}
            make_read(25000, seq),    //
            make_read(27500, seq_rc)  // truncated reverse complement of {5}
    };

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 5, messages);
    // one thread, one read - force reads through in order
    pipeline_desc.add_node<dorado::PairingNode>(
            {sink},
            dorado::DuplexPairingParameters{dorado::ReadOrder::BY_CHANNEL,
                                            dorado::DEFAULT_DUPLEX_CACHE_DEPTH},
            1, 1);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    for (auto& read : reads) {
        pipeline->push_message(std::move(read));
    }
    pipeline.reset();

    // the 4 split reads generate one additional readpair
    CATCH_CHECK(messages.size() == 9);
    auto num_reads =
            std::count_if(messages.begin(), messages.end(), [](const dorado::Message& message) {
                return std::holds_alternative<dorado::SimplexReadPtr>(message);
            });
    CATCH_CHECK(num_reads == 7);
    auto num_pairs =
            std::count_if(messages.begin(), messages.end(), [](const dorado::Message& message) {
                return std::holds_alternative<dorado::ReadPair>(message);
            });
    CATCH_CHECK(num_pairs == 2);
}
