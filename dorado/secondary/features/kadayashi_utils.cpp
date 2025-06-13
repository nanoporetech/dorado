#include "secondary/features/kadayashi_utils.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace dorado::secondary {

namespace {

template <typename T>
void read_checked(std::ifstream& ifs,
                  T* buffer,
                  const size_t num_elements,
                  const std::string_view context) {
    const int64_t num_bytes = static_cast<int64_t>(sizeof(T) * num_elements);
    ifs.read(reinterpret_cast<char*>(buffer), num_bytes);
    if (ifs.gcount() != num_bytes) {
        const std::string context_suffix =
                (context.empty() ? "" : " (" + std::string(context) + ")");
        throw std::runtime_error("Failed to read " + std::to_string(num_bytes) +
                                 " bytes from file." + context_suffix);
    }
}

template <typename T>
T read_checked_pod(std::ifstream& ifs, const std::string_view context) {
    T ret{};
    read_checked(ifs, &ret, 1, context);
    return ret;
}

void seek_checked(std::ifstream& ifs,
                  const std::streamoff offset,
                  const std::ios::seekdir dir,
                  const std::string_view context) {
    ifs.seekg(offset, dir);
    if (!ifs) {
        const std::string context_suffix =
                (context.empty() ? "" : " (" + std::string(context) + ")");
        throw std::runtime_error("Seek failed." + context_suffix);
    }
}

}  // namespace

std::unordered_map<std::string, int32_t> query_bin_file_get_qname2tag(
        const std::filesystem::path& in_haplotag_bin_fn,
        const std::string& chrom,
        const int64_t chrom_start,
        const int64_t chrom_end) {
    constexpr bool DEBUG_PRINT = false;

    std::unordered_map<std::string, int32_t> qname2tag;

    if (std::empty(in_haplotag_bin_fn)) {
        throw std::runtime_error{"Empty input path provided to query_bin_file_get_qname2tag."};
    }

    std::ifstream ifs(in_haplotag_bin_fn, std::ios::binary);

    if (!ifs) {
        throw std::runtime_error{"Could not open file: " + in_haplotag_bin_fn.string() +
                                 " for reading!"};
    }

    if ((chrom_start < 0) || (chrom_end <= 0) || (chrom_end <= chrom_start)) {
        spdlog::warn(
                "Input to query_bin_file_get_qname2tag is not valid. chrom_start = {}, chrom_end = "
                "{}. Returning an empty haplotag lookup.",
                chrom_start, chrom_end);
        return {};
    }

    const uint64_t chrom_start_uint = static_cast<uint64_t>(chrom_start);
    const uint64_t chrom_end_uint = static_cast<uint64_t>(chrom_end);

    // Load the number of reference sequences.
    const uint32_t n_ref = read_checked_pod<uint32_t>(ifs, "n_ref");

    // Find the ID of chrom.
    std::string ref_name;
    int32_t ref_i = -1;
    for (uint32_t i = 0; i < n_ref; ++i) {
        // Header length.
        const uint8_t tn_l = read_checked_pod<uint8_t>(ifs, "tn_l");

        if (ref_i < 0) {  // haven't found the chrom yet
            // Load the ref name.
            ref_name.resize(tn_l);
            read_checked(ifs, ref_name.data(), tn_l, "ref_name");

            // Found the ref.
            if (ref_name == chrom) {
                if (DEBUG_PRINT) {
                    spdlog::debug("[dbg::{}] found ref {}\n", std::string(__func__), ref_name);
                }
                ref_i = static_cast<int32_t>(i);
            }
        } else {
            // Skip other ref names after we found chrom.
            seek_checked(ifs, tn_l, std::ios::cur, "tn_l");
        }
    }

    // Sanity check.
    if (ref_i < 0) {
        spdlog::warn(
                "Reference '{}' not found in the input haplotag bin file: '{}'! Returning an empty "
                "haplotag lookup.",
                chrom, in_haplotag_bin_fn.string());
        return {};
    }

    // Find info for this reference.
    seek_checked(ifs, ref_i * (sizeof(uint64_t) + sizeof(uint32_t)), std::ios::cur, "reference");

    // Load the position of the intervals and the number of intervals.
    const uint64_t pos_intervals_start = read_checked_pod<uint64_t>(ifs, "pos_intervals_start");
    const uint32_t n_intervals = read_checked_pod<uint32_t>(ifs, "n_intervals");

    // Move to the part of the file with the intervals.
    seek_checked(ifs, pos_intervals_start, std::ios::beg, "pos_intervals_start");

    // Search to find the ID of a fulfilling chunk.
    uint64_t best_ovlp_len = 0;  // will check all overlapping intervals and take
                                 // the one with largest ovlp. Tie break is arbitrary
                                 // though stable wrt the bin file.
    uint64_t best_ovlp_chunk_start = 0;
    uint32_t best_ovlp_nreads = 0;
    uint32_t best_ovlp_start = 0;
    uint32_t best_ovlp_end = 0;

    for (uint32_t i = 0; i < n_intervals; ++i) {
        const uint32_t start = read_checked_pod<uint32_t>(ifs, "start");
        const uint32_t end = read_checked_pod<uint32_t>(ifs, "end");
        const uint64_t pos_chunk_start = read_checked_pod<uint64_t>(ifs, "pos_chunk_start");
        const uint32_t n_reads = read_checked_pod<uint32_t>(ifs, "n_reads");

        if ((chrom_end_uint > start) && (chrom_start_uint < end)) {
            if (DEBUG_PRINT) {
                spdlog::debug("[dbg::{}] checking {}-{}\n", std::string(__func__), start, end);
            }
            const uint64_t l = std::min(static_cast<uint64_t>(end), chrom_end_uint) -
                               std::max(static_cast<uint64_t>(start), chrom_start_uint);
            if (l > best_ovlp_len) {
                if (DEBUG_PRINT) {
                    spdlog::debug(
                            "[dbg::{}] update best hit to: {}-{} with {} reads (l: {} => {})\n",
                            std::string(__func__), start, end, n_reads, best_ovlp_len, l);
                }
                best_ovlp_len = l;
                best_ovlp_chunk_start = pos_chunk_start;
                best_ovlp_nreads = n_reads;
                best_ovlp_start = start;
                best_ovlp_end = end;
            } else {
                if (DEBUG_PRINT) {
                    spdlog::debug("[dbg::{}] hit (worse): {}-{} with {} reads\n",
                                  std::string(__func__), start, end, n_reads);
                }
            }
        } else if (start > chrom_end_uint) {
            break;
        }
    }

    if (best_ovlp_chunk_start == 0) {
        spdlog::warn(
                "Reference '{}' found, but requested interval not found ({}:{}-{}). Returning an "
                "empty "
                "haplotag lookup.",
                chrom, chrom, chrom_start, chrom_end);
        return {};
    }

    // Move to the found chunk.
    seek_checked(ifs, best_ovlp_chunk_start, std::ios::beg, "best_ovlp_chunk_start");

    if (DEBUG_PRINT) {
        spdlog::debug("[M::{}] use interval {}-{}\n", std::string(__func__), best_ovlp_start,
                      best_ovlp_end);
    }

    // Read the chunk.
    std::string buffer(256, '\0');
    bool found = false;
    for (uint32_t j = 0; j < best_ovlp_nreads; ++j) {
        // Read the query name.
        const uint8_t qn_l = read_checked_pod<uint8_t>(ifs, "qn_l");
        read_checked(ifs, buffer.data(), qn_l, "buffer");
        const std::string qn = buffer.substr(0, qn_l);

        // Read the haplotag.
        const uint8_t haptag = read_checked_pod<uint8_t>(ifs, "haptag");

        if (qname2tag.count(qn) != 0) {
            spdlog::warn("Query name '{}' already seen in the haplotag file. Skipping.", qn);
            continue;
        }

        qname2tag[qn] = haptag;
        found = true;

        if (DEBUG_PRINT) {
            spdlog::debug("[dbg::{}] insert qn '{}' tag '{}'\n", std::string(__func__), qn, haptag);
        }
    }

    if (!found) {
        spdlog::warn(
                "Reference '{}' found, but there are no queries in the requested interval "
                "({}:{}-{}). Returning an empty "
                "haplotag lookup.",
                chrom, chrom, chrom_start, chrom_end);
        return {};
    }

    return qname2tag;
}

}  // namespace dorado::secondary