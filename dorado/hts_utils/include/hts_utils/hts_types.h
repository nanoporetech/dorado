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
    std::string alias = {};
    std::string type = "na";
    std::pair<int, int> top_barcode_pos = {-1, -1};
    std::pair<int, int> bottom_barcode_pos = {-1, -1};
    bool found_midstrand = false;
};

struct ReadGroup {
    std::string run_id{};
    std::string basecalling_model{};
    std::string modbase_models{};
    std::string flowcell_id{};
    std::string device_id{};
    std::string exp_start_time{};
    std::string sample_id{};
    std::string position_id{};
    std::string experiment_id{};
    std::string acq_start_time{};
    int model_stride{};
};

class HtsData {
public:
    struct ReadAttributes {
        std::string sequencing_kit{};
        std::string experiment_id{};
        std::string sample_id{"no_sample"};
        std::string position_id{"0"};
        std::string flowcell_id{"UNKNOWN"};
        std::string protocol_run_id{"00000000-0000-0000-0000-000000000000"};
        std::string acquisition_id{"0000000000000000000000000000000000000000"};
        int64_t protocol_start_time_ms{0};
        std::size_t subread_id{0};
        bool is_status_pass{true};
        uint64_t start_time_ms{0};
        int model_stride{-1};
        int num_alignments{0};
        int num_secondary_alignments{0};
        int num_supplementary_alignments{0};

        auto operator<=>(const ReadAttributes&) const = default;
    };

    struct ReadAttributesCoreComparator {
        bool operator()(const dorado::HtsData::ReadAttributes&,
                        const dorado::HtsData::ReadAttributes&) const;
    };

    struct ReadAttributesCoreHasher {
        size_t operator()(const dorado::HtsData::ReadAttributes&) const;

    private:
        template <typename T>
        void hash_combine(std::size_t& seed, const T& value) const;
    };

    BamPtr bam_ptr;
    ReadAttributes read_attrs{};
    std::shared_ptr<BarcodeScoreResult> barcoding_result{};
    PrimerClassification primer_classification{};
    std::pair<int, int> adapter_trim_interval{};
    std::pair<int, int> barcode_trim_interval{};
};

std::string to_string(const HtsData::ReadAttributes& a);

}  // namespace dorado
