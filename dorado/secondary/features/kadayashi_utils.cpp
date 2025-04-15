#include "kadayashi_utils.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace dorado::secondary {

std::unordered_map<std::string, int32_t> query_bin_file_get_qname2tag(
        const std::filesystem::path &in_haplotag_bin_fn,
        const std::string &chrom,
        const int64_t chrom_start,
        const int64_t chrom_end) {
    constexpr bool DEBUG_PRINT = false;

    std::unordered_map<std::string, int32_t> qname2tag;

    if (std::empty(in_haplotag_bin_fn)) {
        throw std::runtime_error{"Empty input path provided to query_bin_file_get_qname2tag."};
    }

    std::unique_ptr<FILE, int (*)(FILE *)> fp(fopen(in_haplotag_bin_fn.string().c_str(), "rb"),
                                              &fclose);

    if (!fp.get()) {
        throw std::runtime_error{"Could not open file: " + in_haplotag_bin_fn.string() +
                                 " for reading!"};
    }

    if ((chrom_start < 0) || (chrom_end <= 0) || (chrom_end <= chrom_start)) {
        spdlog::warn(
                "Input to query_bin_file_get_qname2tag is not valid. chrom_start = {}, chrom_end = "
                "{}. Returning an empty haplotag lookup.",
                chrom_start, chrom_end);
    }

    const uint64_t chrom_start_uint = static_cast<uint64_t>(chrom_start);
    const uint64_t chrom_end_uint = static_cast<uint64_t>(chrom_end);

    // Load the number of reference sequences.
    uint32_t n_ref = 0;
    std::ignore = fread(&n_ref, sizeof(uint32_t), 1, fp.get());

    // Find the ID of chrom.
    std::string ref_name;
    int32_t ref_i = -1;
    for (uint32_t i = 0; i < n_ref; ++i) {
        // Header length.
        uint8_t tn_l = 0;
        std::ignore = fread(&tn_l, 1, 1, fp.get());

        if (ref_i < 0) {  // haven't found the chrom yet
            // Load the ref name.
            ref_name.resize(tn_l);
            std::ignore = fread(ref_name.data(), 1, tn_l, fp.get());

            // Found the ref.
            if (ref_name == chrom) {
                if (DEBUG_PRINT) {
                    spdlog::debug("[dbg::{}] found ref {}\n", std::string(__func__), ref_name);
                }
                ref_i = static_cast<int32_t>(i);
            }
        } else {
            // Skip other ref names after we found chrom.
            std::ignore = fseek(fp.get(), tn_l, SEEK_CUR);
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
    {
        const int32_t rv = fseek(fp.get(), ref_i * (sizeof(uint64_t) + sizeof(uint32_t)), SEEK_CUR);
        if (rv) {
            spdlog::warn(
                    "Could not fseek to the data for chrom: '{}'. Returning an empty haplotag "
                    "lookup.",
                    chrom);
            return {};
        }
    }

    // Load the position of the intervals and the number of intervals.
    uint64_t pos_intervals_start = 0;
    uint32_t n_intervals = 0;
    std::ignore = fread(&pos_intervals_start, sizeof(uint64_t), 1, fp.get());
    std::ignore = fread(&n_intervals, sizeof(uint32_t), 1, fp.get());

    // Move to the part of the file with the intervals.
    fseek(fp.get(), pos_intervals_start, SEEK_SET);

    // Search to find the ID of a fulfilling chunk.
    uint64_t best_ovlp_len = 0;  // will check all overlapping intervals and take
                                 // the one with largest ovlp. Tie break is arbitrary
                                 // though stable wrt the bin file.
    uint64_t best_ovlp_chunk_start = 0;
    uint32_t best_ovlp_nreads = 0;
    uint32_t best_ovlp_start = 0;
    uint32_t best_ovlp_end = 0;

    for (uint32_t i = 0; i < n_intervals; ++i) {
        uint32_t start = 0;
        uint32_t end = 0;
        uint64_t pos_chunk_start = 0;
        uint32_t n_reads = 0;
        std::ignore = fread(&start, sizeof(uint32_t), 1, fp.get());
        std::ignore = fread(&end, sizeof(uint32_t), 1, fp.get());
        std::ignore = fread(&pos_chunk_start, sizeof(uint64_t), 1, fp.get());
        std::ignore = fread(&n_reads, sizeof(uint32_t), 1, fp.get());

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
    fseek(fp.get(), best_ovlp_chunk_start, SEEK_SET);

    if (DEBUG_PRINT) {
        spdlog::debug("[M::{}] use interval {}-{}\n", std::string(__func__), best_ovlp_start,
                      best_ovlp_end);
    }

    // Read the chunk.
    uint8_t qn_l = 0;
    uint8_t haptag = 0;
    std::string buffer(256, '\0');
    bool found = false;
    for (uint32_t j = 0; j < best_ovlp_nreads; ++j) {
        // Read the query name.
        std::ignore = fread(&qn_l, 1, 1, fp.get());
        std::ignore = fread(buffer.data(), qn_l, 1, fp.get());
        const std::string qn = buffer.substr(0, qn_l);

        // Read the haplotag.
        std::ignore = fread(&haptag, 1, 1, fp.get());

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