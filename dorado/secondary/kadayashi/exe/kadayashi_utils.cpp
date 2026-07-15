#include "kadayashi_utils.h"

#include "secondary/kadayashi/types.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <sstream>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace kadayashi {

namespace {

struct lite_chunk_info_t {
    uint32_t ID;
    uint32_t start, end;
    uint32_t storage_size;
    uint32_t storage_n;  // aka # reads tagged.
                         // double negate this to check if chunk is phased
    uint64_t start_pos_in_bin;
};
}  // namespace

std::vector<std::string> split_deli_line(const std::string &line, const char delimeter) {
    std::vector<std::string> fields;
    std::istringstream ss(line);
    std::string field;
    while (std::getline(ss, field, delimeter)) {
        fields.push_back(field);
    }
    return fields;
}

void write_binary_given_tsv(const std::filesystem::path &fn_tsv,
                            const std::filesystem::path &fn_bin) {
    // selfnote for first serialize impl:
    // tsv must have been written sorted: it iterated through
    // references, and slided from left to right in each reference,
    // and the chunkID is 0-index integer that is continuously incremental
    // through this process.

    std::ifstream fp_tsv(fn_tsv);
    assert(fp_tsv);
    std::ofstream fp_bin(fn_bin, std::ios::binary);
    assert(fp_bin);

    // for indexing
    std::vector<char> refnames;
    std::vector<uint64_t> ref2chunkIDrange;
    std::vector<lite_chunk_info_t> chunkinfos;
    uint32_t tot_refs = 0;

    uint64_t chunkID = 0;
    std::string line;
    std::string last_refname = "";
    for (int pass = 0; pass < 2; pass++) {
        fp_tsv.clear();
        fp_tsv.seekg(0, std::ios::beg);
        while (std::getline(fp_tsv, line)) {
            if (line.empty()) {
                continue;
            }

            const std::vector<std::string> cols = split_deli_line(line, '\t');

            if (cols[0] == "C" && pass == 0) {  // encountered a new chunk block
                chunkinfos.push_back(
                        {.ID = static_cast<uint32_t>(chunkID),
                         .start = static_cast<uint32_t>(strtoul(cols[3].c_str(), NULL, 10)),
                         .end = static_cast<uint32_t>(strtoul(cols[4].c_str(), NULL, 10)),
                         .storage_size = 0,
                         .storage_n = 0,
                         .start_pos_in_bin = std::numeric_limits<uint64_t>::max()});
                const std::string &refname = cols[2];
                if (last_refname.empty() || refname != last_refname) {
                    last_refname = refname;
                    refnames.push_back((char)refname.size());  // lead a refname with its length
                    for (const char c : refname) {
                        refnames.push_back(c);
                    }
                    ref2chunkIDrange.push_back(chunkID << 32 | (chunkID + 1));
                    tot_refs++;
                } else {
                    ref2chunkIDrange.back()++;
                }
                chunkID++;
            } else if (cols[0] == "R" && pass == 0) {  // collect size of storages
                const std::string &qn = cols[2];
                if (qn.size() > std::numeric_limits<uint8_t>::max()) {
                    spdlog::error("[kdys::{}] read name too long (l={})\n", __func__,
                                  (int)qn.size());
                    exit(1);
                }
                int size = sizeof(uint8_t) /*length of qname*/ +
                           static_cast<int>(qn.size()) /*qname string, no null term*/ +
                           1 /*uint8_t haptag*/;
                chunkinfos.back().storage_n += 1;
                chunkinfos.back().storage_size += size;
            } else if (cols[0] == "R" && pass == 1) {  // write chunk contents to storage
                const std::string &qn = cols[2];
                uint8_t qname_l = static_cast<uint8_t>(qn.size());
                uint8_t haptag = static_cast<uint8_t>(atoi(cols[3].c_str()));
                fp_bin.write(reinterpret_cast<const char *>(&qname_l), 1);
                fp_bin.write(qn.c_str(), qname_l);
                fp_bin.write(reinterpret_cast<const char *>(&haptag), 1);
            }
        }

        // if nothing is collected, return now
        if (chunkID == 0) {
            fp_tsv.close();
            fp_bin.close();
            return;
        }

        // if first pass is done, write the header part of the binary file now
        // and let second pass write storages.
        if (pass == 0) {
            LOG_TRACE(
                    "[kdys::{}] to write header: {} refs, {} chunks (sancheck: chunkinfo buf "
                    "length "
                    "{})\n",
                    __func__, tot_refs, static_cast<int>(chunkID),
                    static_cast<int>(chunkinfos.size()));
            uint32_t header_offset1 =
                    static_cast<uint32_t>(sizeof(uint32_t)) +
                    static_cast<uint32_t>(refnames.size())  // tot_refs and refnames_s_concat
                    +
                    tot_refs *
                            static_cast<uint32_t>((
                                    sizeof(uint64_t) +
                                    sizeof(uint32_t)));  // for each ref, start as-in-file position of chunk interval info, and how many intervals are there
            uint32_t header_offset2 =
                    static_cast<uint32_t>(chunkID) *
                    static_cast<uint32_t>(
                            sizeof(uint32_t) * 2 + sizeof(uint64_t) +
                            sizeof(uint32_t));  // chunk interval infos: ref_start, ref_end, start as-in-file position of the chunk, and how many entries (qname-haptag pairs) in chunk

            // collect start_pos_in_bin
            chunkinfos[0].start_pos_in_bin = header_offset1 + header_offset2;
            for (uint64_t ic = 1; ic < chunkID; ic++) {
                chunkinfos[ic].start_pos_in_bin =
                        chunkinfos[ic - 1].start_pos_in_bin + chunkinfos[ic - 1].storage_size;
            }

            // write headers
            //(refnames)
            fp_bin.write(reinterpret_cast<const char *>(&tot_refs), sizeof(uint32_t));
            fp_bin.write(refnames.data(), refnames.size());
            // (how to jump to chunk interval infos for each reference)
            for (uint32_t i_ref = 0; i_ref < tot_refs; i_ref++) {
                uint32_t chunkID_start = ref2chunkIDrange[i_ref] >> 32;
                uint64_t chunkID_start_infile =
                        header_offset1 +
                        chunkID_start * (sizeof(uint32_t) * 3 + sizeof(uint64_t) * 1);
                uint32_t chunkID_n = (uint32_t)ref2chunkIDrange[i_ref] - chunkID_start;
                fp_bin.write(reinterpret_cast<const char *>(&chunkID_start_infile),
                             sizeof(uint64_t));
                fp_bin.write(reinterpret_cast<const char *>(&chunkID_n), sizeof(uint32_t));
            }
            fp_bin.flush();
            assert(fp_bin.tellp() == (header_offset1));
            // (chunk interval infos: ref_start, ref_end, start_pos_in_bin, storage_n)
            for (const lite_chunk_info_t &info : chunkinfos) {
                const uint32_t ref_s = info.start;
                const uint32_t ref_e = info.end;
                const uint64_t pos_infile = info.start_pos_in_bin;
                const uint32_t l_inchunk = info.storage_n;  // count, not bytes
                fp_bin.write(reinterpret_cast<const char *>(&ref_s), sizeof(uint32_t));
                fp_bin.write(reinterpret_cast<const char *>(&ref_e), sizeof(uint32_t));
                fp_bin.write(reinterpret_cast<const char *>(&pos_infile), sizeof(uint64_t));
                fp_bin.write(reinterpret_cast<const char *>(&l_inchunk), sizeof(uint32_t));
            }
            fp_bin.flush();
            assert(fp_bin.tellp() == (header_offset1 + header_offset2));
        }  // end of header write
    }
    spdlog::info("[kdys::{}] wrote bin file\n", __func__);

    fp_tsv.close();
    fp_bin.close();
}

std::unordered_map<std::string, int> query_bin_file_get_qname2hp(
        const std::filesystem::path &fn_bin,
        const std::string &chrom,
        const uint32_t ref_start,
        const uint32_t ref_end) {
    int silent = 0;
    int debug_print = 0;
    std::unordered_map<std::string, int> qname2hp;
    assert(!fn_bin.native().empty());
    FILE *fp = fopen(fn_bin.string().c_str(), "rb");

    uint32_t n_ref;
    size_t fret = fread(&n_ref, sizeof(uint32_t), 1, fp);
    if (fret != 1) {
        spdlog::error("[kdys::{}] fread failed", __func__);
        exit(1);
    }

    // get chrom index
    std::string refname;
    int ref_i = -1;
    for (uint32_t i = 0; i < n_ref; i++) {
        uint8_t tn_l;
        fret = fread(&tn_l, 1, 1, fp);
        if (fret != 1) {
            spdlog::error("[kdys::{}] fread failed", __func__);
            exit(1);
        }
        if (ref_i < 0) {  // haven't found the chrom yet
            refname.resize(tn_l);
            fret = fread(&refname[0], 1, tn_l, fp);
            if (fret != tn_l) {
                spdlog::error("[kdys::{}] fread failed", __func__);
                exit(1);
            }
            if (refname == chrom) {  // done; don't break
                if (debug_print) {
                    LOG_TRACE("[kdys::{}] found ref {}\n", __func__, refname);
                }
                ref_i = i;
            }
        } else {
            fseek(fp, tn_l, SEEK_CUR);
        }
    }
    if (ref_i >= 0) {
        int found = 0;
        uint64_t pos_intervals_start;
        uint32_t n_intervals;
        fseek(fp, ref_i * (sizeof(uint64_t) + sizeof(uint32_t)), SEEK_CUR);

        fret = fread(&pos_intervals_start, sizeof(uint64_t), 1, fp);
        fret += fread(&n_intervals, sizeof(uint32_t), 1, fp);
        if (fret != 2) {
            spdlog::error("[kdys::{}] fread failed", __func__);
            exit(1);
        }
        fret = fseek(fp, pos_intervals_start, SEEK_SET);

        // search to find the ID of a fulfilling chunk.
        uint32_t start, end, n_reads;
        uint64_t pos_chunk_start;
        uint32_t best_ovlp_len = 0;  // will check all overlapping intervals and take
                                     // the one with largest ovlp. Tie break is arbitrary
                                     // though stable wrt the bin file.
        uint64_t best_ovlp_chunk_start = 0;
        uint32_t best_ovlp_nreads = 0;
        uint32_t best_ovlp_start = 0, best_ovlp_end = 0;
        for (uint32_t i = 0; i < n_intervals; i++) {
            fret = fread(&start, sizeof(uint32_t), 1, fp);
            fret += fread(&end, sizeof(uint32_t), 1, fp);
            fret += fread(&pos_chunk_start, sizeof(uint64_t), 1, fp);
            fret += fread(&n_reads, sizeof(uint32_t), 1, fp);
            if (fret != 4) {
                spdlog::error("[kdys::{}] fread failed", __func__);
                exit(1);
            }
            if (ref_end >= start && ref_start < end) {
                if (debug_print) {
                    LOG_TRACE("[kdys::{}] checking {}-{}\n", __func__, start, end);
                }
                uint32_t l = 0;
                if (start < ref_start) {
                    l = end > ref_end ? ref_end - ref_start : end - ref_start;
                } else {
                    l = end > ref_end ? ref_end - start : end - start;
                }
                if (l > best_ovlp_len) {
                    if (debug_print) {
                        LOG_TRACE(
                                "[kdys::{}] update best hit to: {}-{} with {} reads (l: {} => "
                                "{})\n",
                                __func__, start, end, n_reads, (int)best_ovlp_len, (int)l);
                    }
                    best_ovlp_len = l;
                    best_ovlp_chunk_start = pos_chunk_start;
                    best_ovlp_nreads = n_reads;
                    best_ovlp_start = start;
                    best_ovlp_end = end;
                } else {
                    if (debug_print) {
                        LOG_TRACE("[kdys::{}] hit (worse): {}-{} with {} reads\n", __func__, start,
                                  end, n_reads);
                    }
                }
            } else if (start > ref_end) {
                break;
            }
        }
        if (best_ovlp_chunk_start > 0) {
            fseek(fp, best_ovlp_chunk_start, SEEK_SET);
            spdlog::info("[kdys::{}] use interval {}-{}\n", __func__, best_ovlp_start,
                         best_ovlp_end);

            // read the chunk
            uint8_t qn_l, haptag;
            std::string qn;
            qn.resize(255);
            for (uint32_t j = 0; j < best_ovlp_nreads; j++) {
                fret = fread(&qn_l, 1, 1, fp);

                if ((uint32_t)(qn_l + 1) > qn.capacity()) {
                    qn.resize(qn_l + 1);
                }
                fret += fread(&qn[0], qn_l, 1, fp);
                qn.resize(qn_l);

                fret += fread(&haptag, 1, 1, fp);
                if (fret != 3) {
                    spdlog::error("[kdys::{}] fread failed", __func__);
                    exit(1);
                }

                if (qname2hp.find(qn) != qname2hp.end()) {
                    if (debug_print) {
                        LOG_TRACE("[kdys::{}] qn {} already seen\n", __func__, qn);
                    }
                } else {
                    qname2hp[qn] = haptag;
                    if (debug_print > 1) {
                        LOG_TRACE("[kdys::{}] insert qn {} tag {}\n", __func__, qn, haptag);
                    }
                }
            }
            found = 1;
        }
        if (!found) {
            spdlog::warn("[kdys::{}] ref found, but requested interval not found ({}:{}-{})\n",
                         __func__, chrom, ref_start, ref_end);
        }
    } else {
        if (!silent) {
            spdlog::warn("[kdys::{}] ref {} not found in bin's header\n", __func__, chrom);
        }
    }

    fclose(fp);
    return qname2hp;
}
}  // namespace kadayashi
