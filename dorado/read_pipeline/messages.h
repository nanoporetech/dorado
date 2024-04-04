#pragma once

#include "models/kits.h"
#include "utils/types.h"

#include <ATen/core/TensorBody.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace dorado {

namespace details {

struct Attributes {
    uint32_t mux{std::numeric_limits<uint32_t>::max()};  // Channel mux
    int32_t read_number{-1};     // Per-channel number of each read as it was acquired by minknow
    int32_t channel_number{-1};  //Channel ID
    std::string start_time{};    //Read acquisition start time
    std::string fast5_filename{};
    uint64_t num_samples;
    // Indicates if this read had end reason `mux_change` or `unblock_mux_change`
    bool is_end_reason_mux_change{false};
};

}  // namespace details

class ClientInfo;

class ReadCommon {
public:
    at::Tensor raw_data;  // Loaded from source file

    int model_stride{-1};  // The down sampling factor of the model

    std::string read_id;                  // Unique read ID (UUID4)
    std::string seq;                      // Read basecall
    std::string qstring;                  // Read Qstring (Phred)
    std::vector<uint8_t> moves;           // Move table
    std::vector<uint8_t> base_mod_probs;  // Modified base probabilities
    std::string run_id;                   // Run ID - used in read group
    std::string flowcell_id;    // Flowcell ID - used in read group and for sample sheet aliasing
    std::string position_id;    // Position ID - used for sample sheet aliasing
    std::string experiment_id;  // Experiment ID - used for sample sheet aliasing
    std::string model_name;     // Read group

    dorado::details::Attributes attributes;

    uint64_t start_time_ms;

    std::shared_ptr<const AdapterInfo> adapter_info;
    std::shared_ptr<const BarcodingInfo> barcoding_info;
    std::shared_ptr<BarcodeScoreResult> barcoding_result;
    std::size_t pre_trim_seq_length{};
    std::pair<int, int> adapter_trim_interval{};
    std::pair<int, int> barcode_trim_interval{};
    std::string alignment_string{};

    // A unique identifier for each input read
    // Split (duplex) reads have the read_tag of the parent (template) and their own subread_id
    uint64_t read_tag{0};

    // Contains information about the client to which this read belongs, e.g includes the client ID.
    // By default it's a standalone implementation which has -1 as the id
    std::shared_ptr<ClientInfo> client_info;

    uint32_t mean_qscore_start_pos = 0;

    float calculate_mean_qscore() const;

    std::vector<BamPtr> extract_sam_lines(bool emit_moves,
                                          uint8_t modbase_threshold,
                                          bool is_duplex_parent) const;

    // Barcode.
    std::string barcode{};

    uint64_t sample_rate;  // Loaded from source file

    float shift;                 // To be set by scaler
    float scale;                 // To be set by scaler
    std::string scaling_method;  // To be set by scaler
    std::string parent_read_id;  // Origin read ID for all its subreads. Empty for nonsplit reads.

    std::shared_ptr<const ModBaseInfo>
            mod_base_info;  // Modified base settings of the models that ran on this read

    uint64_t num_trimmed_samples;  // Number of samples which have been trimmed from the raw read.

    bool is_duplex{false};

    size_t get_raw_data_samples() const { return is_duplex ? raw_data.size(1) : raw_data.size(0); }

    // `True` if the basecall model is an RNA model
    bool is_rna_model{false};

    // The chemistry type if any
    models::Chemistry chemistry{models::Chemistry::UNKNOWN};
    // The rapid chemistry adapter type if any - sourced from the read info data
    models::RapidChemistry rapid_chemistry{models::RapidChemistry::UNKNOWN};

    // Track length of estimated polyA tail in bases.
    int rna_poly_tail_length{-1};
    // Track position of end of RNA adapter in signal space. If the RNA adapter is
    // trimmed, this will be 0. Otherwise it will be the position in the signal
    // where the adapter ends.
    int rna_adapter_end_signal_pos{0};

    // subread_id is used to track 2 types of offsprings of a read
    // (1) read splits
    // (2) duplex pairs which share this read as the template read
    size_t subread_id{0};
    size_t split_count{1};
    uint32_t split_point{0};

    // Metadata used by basecall server.
    float model_q_bias{0.0f};
    float model_q_scale{0.0f};

private:
    void generate_duplex_read_tags(bam1_t*) const;
    void generate_read_tags(bam1_t* aln, bool emit_moves, bool is_duplex_parent) const;
    void generate_modbase_tags(bam1_t* aln, uint8_t threshold) const;
    std::string generate_read_group() const;
};

// Class representing a duplex read, including stereo-encoded raw data
class DuplexRead {
public:
    // Data used to generate the stereo features in read_common.raw_data.
    class StereoFeatureInputs {
    public:
        std::vector<unsigned char> alignment;
        uint64_t template_seq_start = std::numeric_limits<uint64_t>::max();
        uint64_t complement_seq_start = std::numeric_limits<uint64_t>::max();
        std::string template_seq;
        std::string complement_seq;
        std::string template_qstring;
        std::string complement_qstring;
        std::vector<uint8_t> template_moves;
        std::vector<uint8_t> complement_moves;
        at::Tensor template_signal;
        at::Tensor complement_signal;
        int signal_stride = -1;
    };
    StereoFeatureInputs stereo_feature_inputs;

    ReadCommon read_common;
};

// Class representing a simplex read, including raw data
class SimplexRead {
public:
    ReadCommon read_common;

    float digitisation;  // Loaded from source file
    float range;         // Loaded from source file
    float offset;        // Loaded from source file

    uint64_t get_end_time_ms() const;

    float scaling;  // Scale factor applied to convert raw integers from sequencer into pore current values

    uint64_t start_sample;
    uint64_t end_sample;
    uint64_t run_acquisition_start_time_ms;
    // Calculate mean Q-score from this position onwards if read is
    // a short read.

    std::atomic_size_t num_duplex_candidate_pairs{0};

    // This is atomic because multiple threads can write to it at the same time.
    // For example, if a read (call it 2) is in the cache, and is selected as a potential pair match by two incoming reads (1 and 3) on two other threads, these threads can both update `is_duplex_parent` at the same time.
    std::atomic_bool is_duplex_parent{false};

    // Track the previous/next read fom the same channel/mux.
    std::string prev_read;
    std::string next_read;
};

using SimplexReadPtr = std::unique_ptr<SimplexRead>;
using DuplexReadPtr = std::unique_ptr<DuplexRead>;

// A pair of reads for Duplex calling
struct ReadPair {
    struct ReadData {
        ReadCommon read_common;
        uint64_t seq_start;
        uint64_t seq_end;
        static ReadData from_read(const SimplexRead& read, uint64_t seq_start, uint64_t seq_end);
    };
    ReadData template_read;
    ReadData complement_read;
};

class CacheFlushMessage {
public:
    int32_t client_id;
};

// The Message type is a std::variant that can hold different types of message objects.
// It is currently able to store:
// - a SimplexReadPtr object, which represents a single Simplex read
// - a DuplexReadPtr object, which represents a single Duplex read
// - a BamPtr object, which represents a raw BAM alignment record
// - a ReadPair object, which represents a pair of reads for duplex calling
// To add more message types, simply add them to the list of types in the std::variant.
using Message = std::variant<SimplexReadPtr, BamPtr, ReadPair, CacheFlushMessage, DuplexReadPtr>;

bool is_read_message(const Message& message);

ReadCommon& get_read_common_data(const Message& message);

// Ensures the raw_data field is non-empty, which it won't necessarily be for DuplexRead.
void materialise_read_raw_data(Message& message);

}  // namespace dorado
