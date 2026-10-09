#pragma once

#include "hts_utils/hts_types.h"
#include "models/kits.h"

#include <ATen/core/TensorBody.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

class ClientInfo;
struct AlignmentResult;
struct ModBaseInfo;

namespace details {

// Note: update StereoDuplexEncoderNode when adding to this struct.
struct Attributes {
    uint32_t mux{std::numeric_limits<uint32_t>::max()};  // Channel mux
    int32_t read_number{-1};     // Per-channel number of each read as it was acquired by minknow
    int32_t channel_number{-1};  // Channel ID
    std::string start_time{};    // Read acquisition start time
    std::string filename{};
    // Indicates if this read had end reason `mux_change` or `unblock_mux_change`
    bool is_end_reason_mux_change{false};
    std::string end_reason{"unknown"};
    std::string pore_type{"not_set"};
    // Loaded from source file.
    uint64_t sample_rate{0};
    int model_stride{-1};  // The down sampling factor of the model
};

}  // namespace details

class ReadCommon {
public:
    explicit ReadCommon();
    ReadCommon(const ReadCommon&);
    ReadCommon& operator=(const ReadCommon&);
    ReadCommon(ReadCommon&&) noexcept;
    ReadCommon& operator=(ReadCommon&&) noexcept;
    ~ReadCommon();

    static constexpr int POLY_TAIL_NOT_FOUND = -1;
    static constexpr int POLY_TAIL_NOT_ENABLED = -2;
    at::Tensor raw_data;  // Loaded from source file

    /*
    Note: Update read_utils shallow_copy_read to ensure split reads copy all fields,
          and StereoDuplexEncoderNode for duplex reads.
    */
    std::string read_id;                            // Unique read ID (UUID4)
    std::string seq;                                // Read basecall
    std::string qstring;                            // Read Qstring (Phred)
    std::vector<uint8_t> moves;                     // Move table
    std::vector<uint8_t> base_mod_probs;            // Modified base probabilities
    std::vector<bool> base_mod_simplex_motif_hits;  // Modbase motif hits, simplex only
    std::string run_id;                             // Run ID - used in read group
    std::string flow_cell_product_code;             // Flowcell product code
    std::string sequencing_kit;  // Sequencing kit - Used in primer detection/classification
    std::string flowcell_id;     // Flowcell ID - used in read group and for sample sheet aliasing
    std::string position_id;     // Position ID - used for sample sheet aliasing
    std::string acquisition_id;  // Acquisition ID - IDs an acquisition within a protocol.
    std::string experiment_id;   // Experiment ID - used in aliasing
    std::string model_name;      // Read group
    std::string sample_id;       // User-supplied name for the sample being analysed.
    int64_t protocol_start_time_ms{0};  // Start time of the protocol
    uint64_t num_minknow_events{0};     // Number of minknow events or zero if unknown

    dorado::details::Attributes attributes;

    uint64_t start_time_ms = 0;

    std::shared_ptr<BarcodeScoreResult> barcoding_result;
    PrimerClassification primer_classification{};
    // NB: refers to read length before adapter/primer/barcode trimming, but after mux change trim or read splitting
    std::size_t pre_trim_seq_length{};
    std::pair<int, int> adapter_trim_interval{};
    std::pair<int, int> barcode_trim_interval{};
    std::vector<AlignmentResult> alignment_results;

    // A unique identifier for each input read
    // Split (duplex) reads have the read_tag of the parent (template) and their own subread_id
    uint64_t read_tag{0};

    // Contains information about the client to which this read belongs, e.g includes the client ID.
    // By default it's a standalone implementation which has -1 as the id
    std::shared_ptr<ClientInfo> client_info;

    uint32_t mean_qscore_start_pos = 0;

    float calculate_mean_qscore() const;

    std::vector<BamPtr> extract_sam_lines(bool emit_moves,
                                          std::optional<uint8_t> modbase_threshold,
                                          bool is_duplex_parent) const;

    // Barcode.
    std::string barcode{};

    float shift = 0;             // To be set by scaler
    float scale = 0;             // To be set by scaler
    std::string scaling_method;  // To be set by scaler
    std::string parent_read_id;  // Origin read ID for all its subreads. Empty for nonsplit reads.

    std::shared_ptr<const ModBaseInfo>
            mod_base_info;  // Modified base settings of the models that ran on this read

    // Number of samples which have been trimmed from the raw read.
    uint64_t num_trimmed_samples = 0;

    bool is_duplex{false};

    size_t get_raw_data_samples() const { return is_duplex ? raw_data.size(1) : raw_data.size(0); }

    // `True` if the basecall model is an RNA model
    bool is_rna_model{false};

    // The chemistry type if any
    models::Chemistry chemistry{models::Chemistry::UNKNOWN};
    // The rapid chemistry adapter type if any - sourced from the read info data
    models::RapidChemistry rapid_chemistry{models::RapidChemistry::UNKNOWN};

    // Track length of estimated polyA tail in bases.
    int rna_poly_tail_length{POLY_TAIL_NOT_ENABLED};
    // Raw signal position used as the anchor for the poly tail search.
    int poly_tail_signal_anchor{POLY_TAIL_NOT_ENABLED};
    // Estimated ranges for the poly tail signal. This may have multiple entries (for plasmids) if a split tail is detected
    std::array<std::pair<int, int>, 2> poly_tail_signal_boundaries;

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

    TrimFlags trim_flags{};

private:
    void generate_duplex_read_tags(bam1_t*) const;
    void generate_read_tags(bam1_t* aln, bool emit_moves, bool is_duplex_parent) const;
    void generate_modbase_tags(bam1_t* aln, std::optional<uint8_t> threshold) const;
    std::string generate_read_group() const;
};

}  // namespace dorado
