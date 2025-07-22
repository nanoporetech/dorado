#pragma once

#include <memory>
#include <stdexcept>
#include <string>

struct bam1_t;
struct sam_hdr_t;
struct htsFile;

namespace dorado {

inline const std::string UNCLASSIFIED_STR = "unclassified";

struct BamDestructor {
    void operator()(bam1_t*);
};
using BamPtr = std::unique_ptr<bam1_t, BamDestructor>;

struct SamHdrDestructor {
    void operator()(sam_hdr_t*);
};
using SamHdrPtr = std::unique_ptr<sam_hdr_t, SamHdrDestructor>;

class SamHdrSharedPtr {
public:
    explicit SamHdrSharedPtr(sam_hdr_t* hdr_ptr);
    explicit SamHdrSharedPtr(SamHdrPtr hdr);

    SamHdrSharedPtr(const std::shared_ptr<const sam_hdr_t>&) = delete;

    const sam_hdr_t* get() const { return m_header.get(); }
    const sam_hdr_t& operator*() const { return *m_header; }
    const sam_hdr_t* operator->() const { return m_header.get(); }

    std::shared_ptr<const sam_hdr_t> ptr() const { return m_header; }

    bool operator==(std::nullptr_t) const noexcept { return m_header == nullptr; }
    bool operator!=(std::nullptr_t) const noexcept { return m_header != nullptr; }

private:
    std::shared_ptr<const sam_hdr_t> m_header;
};

struct HtsFileDestructor {
    void operator()(htsFile*);
};
using HtsFilePtr = std::unique_ptr<htsFile, HtsFileDestructor>;

enum class StrandOrientation : int {
    REVERSE = -1,  ///< "-" orientation
    UNKNOWN = 0,   ///< "?" orientation
    FORWARD = 1,   ///< "+" orientation
};

inline char to_char(const StrandOrientation orientation) {
    switch (orientation) {
    case StrandOrientation::REVERSE:
        return '-';
    case StrandOrientation::FORWARD:
        return '+';
    case StrandOrientation::UNKNOWN:
        return '?';
    default:
        throw std::runtime_error("Invalid orientation value " + std::to_string(int(orientation)));
    }
}

struct PrimerClassification {
    std::string primer_name = UNCLASSIFIED_STR;
    std::string umi_tag_sequence{};
    StrandOrientation orientation = StrandOrientation::UNKNOWN;
};

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
    std::string barcode_name = UNCLASSIFIED_STR;
    std::string kit = UNCLASSIFIED_STR;
    std::string barcode_kit = UNCLASSIFIED_STR;
    std::string variant = "n/a";
    std::pair<int, int> top_barcode_pos = {-1, -1};
    std::pair<int, int> bottom_barcode_pos = {-1, -1};
    bool found_midstrand = false;
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

class HtsData {
public:
    struct ReadAttributes {
        std::string sequencing_kit{};
        std::string experiment_id{};
        std::string sample_id{};
        std::string position_id{};
        std::string flowcell_id{};
        std::string protocol_run_id{};
        std::string acquisition_id{};
        int64_t protocol_start_time_ms{0};
        std::size_t subread_id{0};
    };

    BamPtr bam_ptr;
    ReadAttributes read_attrs{};
    std::shared_ptr<BarcodeScoreResult> barcoding_result{};
    PrimerClassification primer_classification{};
    std::pair<int, int> adapter_trim_interval{};
    std::pair<int, int> barcode_trim_interval{};
};

inline bool operator==(const dorado::HtsData::ReadAttributes& lhs,
                       const dorado::HtsData::ReadAttributes& rhs) {
    // clang-format off
    return lhs.sequencing_kit == rhs.sequencing_kit &&
           lhs.experiment_id == rhs.experiment_id &&
           lhs.sample_id == rhs.sample_id &&
           lhs.position_id == rhs.position_id &&
           lhs.flowcell_id == rhs.flowcell_id &&
           lhs.acquisition_id == rhs.acquisition_id &&
           lhs.protocol_start_time_ms == rhs.protocol_start_time_ms &&
           lhs.subread_id == rhs.subread_id;
    // clang-format on
}

}  // namespace dorado

namespace std {
template <>
struct hash<dorado::HtsData::ReadAttributes> {
    std::size_t operator()(const dorado::HtsData::ReadAttributes& attr) const {
        std::size_t seed = 0;
        hash_combine(seed, attr.sequencing_kit);
        hash_combine(seed, attr.experiment_id);
        hash_combine(seed, attr.sample_id);
        hash_combine(seed, attr.position_id);
        hash_combine(seed, attr.flowcell_id);
        hash_combine(seed, attr.acquisition_id);
        hash_combine(seed, attr.protocol_start_time_ms);
        hash_combine(seed, attr.subread_id);
        return seed;
    }

private:
    template <typename T>
    static void hash_combine(std::size_t& seed, const T& value) {
        seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

}  // namespace std