#include "Barcoder.h"

#include "htslib/sam.h"
#include "minimap.h"
//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include "mmpriv.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

Barcoder::Barcoder(MessageSink& sink, const std::vector<std::string>& barcodes, int threads)
        : MessageSink(10000), m_sink(sink), m_threads(threads) {
    // Initialize option structs.
    mm_set_opt(0, &m_idx_opt, &m_map_opt);
    // Setting options to map-ont default till relevant args are exposed.
    mm_set_opt("map-ont", &m_idx_opt, &m_map_opt);

    mm_check_opt(&m_idx_opt, &m_map_opt);

    int num_barcodes = barcoding::barcodes.size();
    auto seqs = std::make_unique<const char*[]>(num_barcodes);
    auto names = std::make_unique<const char*[]>(num_barcodes);
    int index = 0;
    for (auto& [bc, seq] : barcoding::barcodes) {
        seqs[index] = seq.c_str();
        names[index] = bc.c_str();
        index++;
    }

    m_index = mm_idx_str(10, 15, 0, 4, num_barcodes, seqs.get(), names.get());

    if (mm_verbose >= 3) {
        mm_idx_stat(m_index);
    }

    for (int i = 0; i < m_threads; i++) {
        m_tbufs.push_back(mm_tbuf_init());
    }

    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&Barcoder::worker_thread, this, i)));
    }
}

Barcoder::~Barcoder() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
    for (int i = 0; i < m_threads; i++) {
        mm_tbuf_destroy(m_tbufs[i]);
    }
    mm_idx_destroy(m_index);
    // Adding for thread safety in case worker thread throws exception.
    m_sink.terminate();
}

void Barcoder::worker_thread(size_t tid) {
    m_active++;  // Track active threads.

    Message message;
    while (m_work_queue.try_pop(message)) {
        auto read = std::get<BamPtr>(std::move(message));
        auto records = align(read.get(), m_tbufs[tid]);
        for (auto& record : records) {
            m_sink.push_message(std::move(record));
        }
    }

    int num_active = --m_active;
    if (num_active == 0) {
        terminate();
        m_sink.terminate();
    }
}

// Function to add auxiliary tags to the alignment record.
// These are added to maintain parity with mm2.
void Barcoder::add_tags(bam1_t* record, const mm_reg1_t* aln) {
    std::string bc(m_index->seq[aln->rid].name);
    bam_aux_append(record, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
}

std::vector<BamPtr> Barcoder::align(bam1_t* irecord, mm_tbuf_t* buf) {
    // some where for the hits
    std::vector<BamPtr> results;

    // get the sequence to map from the record
    auto seqlen = irecord->core.l_qseq;

    // get query name.
    std::string_view qname(bam_get_qname(irecord));

    auto bseq = bam_get_seq(irecord);
    std::string seq = utils::convert_nt16_to_str(bseq, seqlen);
    // Pre-generate reverse complement sequence.
    std::string seq_rev = utils::reverse_complement(seq);

    // do the mapping
    int hits = 0;
    mm_reg1_t* reg =
            mm_map(m_index, seq.length(), seq.c_str(), &hits, buf, &m_map_opt, qname.data());

    // just return the input record
    if (hits == 0) {
        std::string bc("unknown");
        bam_aux_append(irecord, "BC", 'Z', bc.length() + 1, (uint8_t*)bc.c_str());
        results.push_back(BamPtr(bam_dup1(irecord)));
    } else {
        spdlog::debug("Found {} hits", hits);
        auto best_map = std::max_element(
                reg, reg + hits,
                [&](const mm_reg1_t& a, const mm_reg1_t& b) { return a.mapq < b.mapq; });

        // new output record
        bam1_t* record = bam_dup1(irecord);

        int32_t tid = best_map->rid;
        hts_pos_t qs = best_map->qs;
        hts_pos_t qe = best_map->qe;
        uint8_t mapq = best_map->mapq;

        if (mapq > 10) {
            // Add new tags to match minimap2.
            add_tags(record, best_map);
        }

        free(best_map->p);
        results.push_back(BamPtr(record));
    }

    free(reg);
    return results;
}

stats::NamedStats Barcoder::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
