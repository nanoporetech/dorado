#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dorado {

class ReadCommon;
class DuplexRead;

namespace modbase {

// Map a context hit index within a duplex strand's realigned sequence onto an index into the
// duplex read sequence.
int64_t resolve_duplex_sequence_index(int64_t hit,
                                      int64_t target_start,
                                      int64_t sequence_size,
                                      bool is_template_direction);

// Presents a simplex read as a single strand whose signal, sequence and moves are those of the read.
class SimplexReadPolicy {
public:
    static constexpr bool is_duplex = false;

    explicit SimplexReadPolicy(ReadCommon& read);

    // Returns true exactly once.
    bool next_strand();

    const at::Tensor& signal() const;
    const std::string& sequence() const;
    const std::vector<uint8_t>& moves() const;
    bool is_template_direction() const { return true; }
    int64_t target_start() const { return 0; }

    // Index into the read sequence of a context hit within the current strand.
    size_t to_read_index(size_t strand_hit) const { return strand_hit; }

    // Record that the read sequence has a motif hit at `strand_hit` of the current strand.
    void mark_motif_hit(size_t strand_hit);

    // Discard all recorded motif hits.
    void reset_motif_hits();

private:
    ReadCommon& m_read;
    bool m_processed{false};
};

// Presents a duplex read as up to two strands (template, then complement). The signal, sequence
// and moves of each strand are realigned to the duplex sequence, and the complement is expressed in
// the template direction. A strand which cannot be realigned is skipped.
class DuplexReadPolicy {
public:
    static constexpr bool is_duplex = true;

    DuplexReadPolicy(DuplexRead& read, size_t canonical_stride);

    // Advance to the next realignable strand, returning false when there are none left.
    bool next_strand();

    const at::Tensor& signal() const { return m_signal; }
    const std::string& sequence() const { return m_sequence; }
    const std::vector<uint8_t>& moves() const { return m_moves; }
    bool is_template_direction() const { return m_is_template_direction; }
    int64_t target_start() const { return m_target_start; }

    // Index into the duplex read sequence of a context hit within the current strand.
    size_t to_read_index(size_t strand_hit) const;

    // Duplex reads do not record motif hits.
    void mark_motif_hit(size_t) const {}
    void reset_motif_hits() const {}

private:
    DuplexRead& m_read;
    const size_t m_canonical_stride;
    int m_strand_index{-1};
    bool m_is_template_direction{true};
    int64_t m_target_start{0};
    at::Tensor m_signal;
    std::string m_sequence;
    std::vector<uint8_t> m_moves;
};

}  // namespace modbase
}  // namespace dorado
