#pragma once

#include <array>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

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

struct SingleEndResult {
    float score = -1.f;
    std::string name = UNCLASSIFIED;
    std::pair<int, int> position = {-1, -1};
};

struct AdapterScoreResult {
    SingleEndResult front;
    SingleEndResult rear;
};

using BarcodeFilterSet = std::optional<std::unordered_set<std::string>>;

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
