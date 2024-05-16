#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

struct bam1_t;
struct htsFile;
struct mm_tbuf_s;
struct sam_hdr_t;

namespace dorado {

struct AdapterInfo {
    bool trim_adapters{true};
    bool trim_primers{true};
    std::optional<std::string> custom_seqs = std::nullopt;
};

struct BarcodingInfo {
    using FilterSet = std::optional<std::unordered_set<std::string>>;
    std::string kit_name{};
    bool barcode_both_ends{false};
    bool trim{false};
    FilterSet allowed_barcodes;
    std::optional<std::string> custom_kit = std::nullopt;
    std::optional<std::string> custom_seqs = std::nullopt;
};

std::shared_ptr<const BarcodingInfo> create_barcoding_info(
        const std::vector<std::string> &kit_names,
        bool barcode_both_ends,
        bool trim_barcode,
        BarcodingInfo::FilterSet allowed_barcodes,
        const std::optional<std::string> &custom_kit,
        const std::optional<std::string> &custom_seqs);

struct BarcodeScoreResult {
    int penalty = -1;
    int top_penalty = -1;
    int bottom_penalty = -1;
    float top_barcode_score = -1.f;
    float bottom_barcode_score = -1.f;
    float barcode_score = -1.f;
    float flank_score = -1.f;
    float top_flank_score = -1.f;
    float bottom_flank_score = -1.f;
    bool use_top = false;
    std::string barcode_name = "unclassified";
    std::string kit = "unclassified";
    std::string barcode_kit = "unclassified";
    std::string variant = "n/a";
    std::pair<int, int> top_barcode_pos = {-1, -1};
    std::pair<int, int> bottom_barcode_pos = {-1, -1};
    bool found_midstrand = false;
};

struct SingleEndResult {
    float score = -1.f;
    std::string name = "unclassified";
    std::pair<int, int> position = {-1, -1};
};

struct AdapterScoreResult {
    SingleEndResult front;
    SingleEndResult rear;
};

struct ReadGroup {
    std::string run_id;
    std::string basecalling_model;
    std::string modbase_models;
    std::string flowcell_id;
    std::string device_id;
    std::string exp_start_time;
    std::string sample_id;
    std::string position_id;
    std::string experiment_id;
};

struct BamDestructor {
    void operator()(bam1_t *);
};
using BamPtr = std::unique_ptr<bam1_t, BamDestructor>;

struct MmTbufDestructor {
    void operator()(mm_tbuf_s *);
};
using MmTbufPtr = std::unique_ptr<mm_tbuf_s, MmTbufDestructor>;

struct SamHdrDestructor {
    void operator()(sam_hdr_t *);
};
using SamHdrPtr = std::unique_ptr<sam_hdr_t, SamHdrDestructor>;

struct HtsFileDestructor {
    void operator()(htsFile *);
};
using HtsFilePtr = std::unique_ptr<htsFile, HtsFileDestructor>;

enum class ReadOrder { UNRESTRICTED, BY_CHANNEL, BY_TIME };

struct DuplexPairingParameters {
    ReadOrder read_order;
    size_t cache_depth;
};
/// Default cache depth to be used for the duplex pairing cache.
constexpr static size_t DEFAULT_DUPLEX_CACHE_DEPTH = 10;

inline std::string to_string(ReadOrder read_order) {
    switch (read_order) {
    case ReadOrder::UNRESTRICTED:
        return "UNRESTRICTED";
    case ReadOrder::BY_CHANNEL:
        return "BY_CHANNEL";
    case ReadOrder::BY_TIME:
        return "BY_TIME";
    default:
        return "Unknown";
    }
}

struct ModBaseInfo {
    ModBaseInfo() = default;
    ModBaseInfo(std::vector<std::string> alphabet_, std::string long_names_, std::string context_)
            : alphabet(std::move(alphabet_)),
              long_names(std::move(long_names_)),
              context(std::move(context_)) {}
    std::vector<std::string> alphabet;
    std::string long_names;
    std::string context;
    std::array<size_t, 4> base_counts{};
};

}  // namespace dorado
