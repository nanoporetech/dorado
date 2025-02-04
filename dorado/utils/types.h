#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct bam1_t;
struct htsFile;
struct mm_tbuf_s;
struct sam_hdr_t;
struct kstring_t;

namespace dorado {

inline const std::string UNCLASSIFIED = "unclassified";

struct AlignmentResult {
    std::string name;    ///< Record name, if applicable
    std::string genome;  ///< Name of the reference genome against which the strand has been aligned
    char direction;  ///< '+' if the alignment matches to the forward strand, '-' if alignment matches the reverse
    int genome_start;  ///< ref[genome_start:genome_end] successfully aligned against strand[strand_start:strand_end]
    int genome_end;  ///< ref[genome_start:genome_end] successfully aligned against strand[strand_start:strand_end]
    int strand_start;  ///< ref[genome_start:genome_end] successfully aligned against strand[strand_start:strand_end]
    int strand_end;  ///< ref[genome_start:genome_end] successfully aligned against strand[strand_start:strand_end]
    int num_events;      ///< Length of input strand
    int num_insertions;  ///< Number of positions in strand that have no counterpart in the reference
    int num_deletions;   ///< Number of positions in genome that have no counterpart in the strand
    int num_aligned;  ///< Equals genome_end - genome_start - num_deletions = strand_end - strand_start - num_insertions
    int num_correct;   ///< Number of aligned positions where bases agree
    float coverage;    ///< num_aligned / min(strand length, reference length)
    float identity;    ///< Equals num_correct / num_aligned
    float accuracy;    ///< Equals num_correct / (num_aligned + num_insertions + num_deletions)
    int strand_score;  ///< Score of the alignment (scale is implementation-specific)
    std::string sam_string;  ///< SAM file contents
    int bed_hits;  ///< How many times this alignment intersects regions of interest in the bed file.
    std::string bed_lines;     ///< list of lines from the bed file which this alignment intersects
    std::string sequence;      ///< The aligned sequence.
    std::string qstring;       ///< The quality string for the aligned sequence.
    bool secondary_alignment;  ///< This is a secondary alignment.
    bool supplementary_alignment;  ///< This is a supplementary alignment.
    int mapping_quality;           ///< MAPQ score

    /// Constructor.
    AlignmentResult()
            : name("unknown"),
              genome("*"),
              direction('*'),
              genome_start(0),
              genome_end(0),
              strand_start(0),
              strand_end(0),
              num_events(-1),
              num_insertions(-1),
              num_deletions(-1),
              num_aligned(-1),
              num_correct(-1),
              coverage(-1.f),
              identity(-1.f),
              accuracy(-1.f),
              strand_score(-1),
              sam_string("*"),
              bed_hits(0),
              bed_lines(""),
              sequence(""),
              qstring(""),
              secondary_alignment(false),
              supplementary_alignment(false),
              mapping_quality(255) {}
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
    std::string barcode_name = UNCLASSIFIED;
    std::string kit = UNCLASSIFIED;
    std::string barcode_kit = UNCLASSIFIED;
    std::string variant = "n/a";
    std::pair<int, int> top_barcode_pos = {-1, -1};
    std::pair<int, int> bottom_barcode_pos = {-1, -1};
    bool found_midstrand = false;
};

struct SingleEndResult {
    float score = -1.f;
    std::string name = UNCLASSIFIED;
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
    std::string primer_name = UNCLASSIFIED;
    std::string umi_tag_sequence{};
    StrandOrientation orientation = StrandOrientation::UNKNOWN;
};

using BarcodeFilterSet = std::optional<std::unordered_set<std::string>>;

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

/// Wrapper for htslib kstring_t struct.
class KString {
public:
    /** Contains an uninitialized kstring_t.
     *  Useful for later assigning to a kstring_t returned by an htslib
     *  API function. If you need to pass object to an API function which
     *  will put data in it, then use the pre-allocate constructor instead,
     *  to avoid library conflicts on windows.
     */
    KString();

    /** Pre-allocate the string with n bytes of storage. If you pass the kstring
     *  into an htslib API function that would resize the kstring_t object as
     *  needed, when the API function does the resize this can result in
     *  strange errors like stack corruption due to differences between the
     *  implementation in the compiled library, and the implementation compiled
     *  into your C++ code using htslib macros. So make sure you pre-allocate
     *  with enough memory to insure that no resizing will be needed.
     */
    KString(size_t n);

    /** This object owns the storage in the internal kstring_t object.
     *  To avoid reference counting, we don't allow this object to be copied.
     */
    KString(const KString &) = delete;

    /** Take ownership of the data in the kstring_t object.
     *  Note that it is an error to create more than one KString object
     *  that owns the same kstring_t data.
     */
    KString(kstring_t &&data) noexcept;

    /// Move Constructor
    KString(KString &&other) noexcept;

    /// No copying allowed.
    KString &operator=(const KString &) = delete;

    /// Move assignment.
    KString &operator=(KString &&rhs) noexcept;

    /// Destroys the kstring_t data.
    ~KString();

    /// Returns the kstring_t object that points to the internal data.
    kstring_t &get() const;

private:
    std::unique_ptr<kstring_t> m_data;
};

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

    // Generate the modbase probability array offsets for the 4 canonical bases
    std::array<size_t, 4> base_probs_offsets() const {
        // Example Mods := 6mA, 5mC, 5hmC
        // base_counts  := [2, 3, 1, 1]
        // probs vector := [A, 6mA, C, 5mC, 5hmC, G, T]
        // offsets      := [0,      2,            5, 6]
        std::array<size_t, 4> offsets;
        offsets[0] = 0;
        offsets[1] = base_counts[0];
        offsets[2] = offsets[1] + base_counts[1];
        offsets[3] = offsets[2] + base_counts[2];
        return offsets;
    }
};

}  // namespace dorado
