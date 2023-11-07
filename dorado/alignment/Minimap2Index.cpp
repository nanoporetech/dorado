#include "Minimap2Index.h"

#include <spdlog/spdlog.h>

//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include <mmpriv.h>

#include <filesystem>

namespace {
struct IndexReaderDeleter {
    void operator()(mm_idx_reader_t* index_reader) { mm_idx_reader_close(index_reader); }
};
using IndexReaderPtr = std::unique_ptr<mm_idx_reader_t, IndexReaderDeleter>;

IndexReaderPtr create_index_reader(const std::string& index_file,
                                   const mm_idxopt_t& index_options) {
    IndexReaderPtr reader;
    reader.reset(mm_idx_reader_open(index_file.c_str(), &index_options, 0));
    return reader;
}

}  // namespace

namespace dorado::alignment {

void Minimap2Index::set_index_options(const Minimap2IndexOptions& index_options) {
    m_index_options.k = index_options.kmer_size;
    m_index_options.w = index_options.window_size;
    spdlog::info("> Index parameters input by user: kmer size={} and window size={}.",
                 m_index_options.k, m_index_options.w);

    m_index_options.batch_size = index_options.index_batch_size;
    m_index_options.mini_batch_size = index_options.index_batch_size;
    spdlog::info("> Index parameters input by user: batch size={} and mini batch size={}.",
                 m_index_options.batch_size, m_index_options.mini_batch_size);
}

void Minimap2Index::set_mapping_options(const Minimap2MappingOptions& mapping_options) {
    m_mapping_options.bw = mapping_options.bandwidth;
    m_mapping_options.bw_long = mapping_options.bandwidth_long;
    spdlog::info("> Map parameters input by user: bandwidth={} and bandwidth long={}.",
                 m_mapping_options.bw, m_mapping_options.bw_long);

    if (!mapping_options.print_secondary) {
        m_mapping_options.flag |= MM_F_NO_PRINT_2ND;
    }
    m_mapping_options.best_n = mapping_options.best_n_secondary;
    spdlog::info(
            "> Map parameters input by user: don't print secondary={} and best n secondary={}.",
            static_cast<bool>(m_mapping_options.flag & MM_F_NO_PRINT_2ND),
            m_mapping_options.best_n);

    if (mapping_options.soft_clipping) {
        m_mapping_options.flag |= MM_F_SOFTCLIP;
    }
    if (mapping_options.secondary_seq) {
        m_mapping_options.flag |= MM_F_SECONDARY_SEQ;
    }
    spdlog::info("> Map parameters input by user: soft clipping={} and secondary seq={}.",
                 static_cast<bool>(m_mapping_options.flag & MM_F_SOFTCLIP),
                 static_cast<bool>(m_mapping_options.flag & MM_F_SECONDARY_SEQ));

    // Force cigar generation.
    m_mapping_options.flag |= MM_F_CIGAR;
}

bool Minimap2Index::load_index_unless_split(const std::string& index_file, int num_threads) {
    auto index_reader = create_index_reader(index_file, m_index_options);
    m_index.reset(mm_idx_reader_read(index_reader.get(), num_threads));
    IndexPtr split_index{};
    split_index.reset(mm_idx_reader_read(index_reader.get(), num_threads));
    if (split_index != nullptr) {
        return false;
    }

    if (m_index->k != m_index_options.k || m_index->w != m_index_options.w) {
        spdlog::warn(
                "Indexing parameters mismatch prebuilt index: using paramateres kmer "
                "size={} and window size={} from prebuilt index.",
                m_index->k, m_index->w);
    }

    if (mm_verbose >= 3) {
        mm_idx_stat(m_index.get());
    }

    return true;
}

IndexLoadResult Minimap2Index::load(const std::string& index_file,
                                    const Minimap2Options& options,
                                    int num_threads) {
    // Check if reference file exists.
    if (!std::filesystem::exists(index_file)) {
        return IndexLoadResult::reference_file_not_found;
    }
    mm_set_opt(0, &m_index_options, &m_mapping_options);
    // Setting options to map-ont default till relevant args are exposed.
    mm_set_opt("map-ont", &m_index_options, &m_mapping_options);

    set_index_options(options);
    set_mapping_options(options);

    if (mm_check_opt(&m_index_options, &m_mapping_options) < 0) {
        return IndexLoadResult::validation_error;
    }

    if (!load_index_unless_split(index_file, num_threads)) {
        return IndexLoadResult::split_index_not_supported;
    }

    mm_mapopt_update(&m_mapping_options, m_index.get());

    if (options.print_aln_seq) {
        mm_dbg_flag |= MM_DBG_PRINT_QNAME | MM_DBG_PRINT_ALN_SEQ;
    }
    spdlog::debug("> Map parameters input by user: dbg print qname={} and aln seq={}.",
                  static_cast<bool>(mm_dbg_flag & MM_DBG_PRINT_QNAME),
                  static_cast<bool>(mm_dbg_flag & MM_DBG_PRINT_ALN_SEQ));

    return IndexLoadResult::success;
}

HeaderSequenceRecords Minimap2Index::get_sequence_records_for_header() const {
    std::vector<std::pair<char*, uint32_t>> records;
    for (int i = 0; i < m_index->n_seq; ++i) {
        records.push_back(std::make_pair(m_index->seq[i].name, m_index->seq[i].len));
    }
    return records;
}

}  // namespace dorado::alignment