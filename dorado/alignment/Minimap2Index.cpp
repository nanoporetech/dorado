#include "Minimap2Index.h"

#include <spdlog/spdlog.h>

//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include <mmpriv.h>

#include <cassert>
#include <cstdio>
#include <filesystem>

namespace {

struct IndexDeleter {
    void operator()(mm_idx_t* index) { mm_idx_destroy(index); }
};
using IndexUniquePtr = std::unique_ptr<mm_idx_t, IndexDeleter>;

dorado::alignment::IndexReaderPtr create_index_reader(const std::string& index_file,
                                                      const mm_idxopt_t& index_options) {
    dorado::alignment::IndexReaderPtr reader;
    reader.reset(mm_idx_reader_open(index_file.c_str(), &index_options, 0));
    return reader;
}

}  // namespace

namespace dorado::alignment {

void IndexReaderDeleter::operator()(mm_idx_reader_t* index_reader) {
    mm_idx_reader_close(index_reader);
}

void Minimap2Index::set_index_options(const Minimap2IndexOptions& index_options) {
    m_index_options->k = index_options.kmer_size.value_or(m_index_options->k);
    m_index_options->w = index_options.window_size.value_or(m_index_options->w);
    spdlog::trace("> Index parameters input by user: kmer size={} and window size={}.",
                  m_index_options->k, m_index_options->w);

    //if not specified override the preset value of 8000000000 with 16000000000
    m_index_options->batch_size = index_options.index_batch_size.value_or(16000000000);
    m_index_options->mini_batch_size = m_index_options->batch_size;
    spdlog::trace("> Index parameters input by user: batch size={} and mini batch size={}.",
                  m_index_options->batch_size, m_index_options->mini_batch_size);
}

void Minimap2Index::set_mapping_options(const Minimap2MappingOptions& mapping_options) {
    m_mapping_options->bw = mapping_options.bandwidth.value_or(m_mapping_options->bw);
    m_mapping_options->bw_long =
            mapping_options.bandwidth_long.value_or(m_mapping_options->bw_long);
    spdlog::trace("> Map parameters: bandwidth={} and bandwidth long={}.", m_mapping_options->bw,
                  m_mapping_options->bw_long);

    if (!mapping_options.print_secondary.value_or(true)) {
        m_mapping_options->flag |= MM_F_NO_PRINT_2ND;
    }
    m_mapping_options->best_n =
            mapping_options.best_n_secondary.value_or(m_mapping_options->best_n);
    spdlog::trace(
            "> Map parameters input by user: don't print secondary={} and best n secondary={}.",
            static_cast<bool>(m_mapping_options->flag & MM_F_NO_PRINT_2ND),
            m_mapping_options->best_n);

    if (mapping_options.soft_clipping.value_or(false)) {
        m_mapping_options->flag |= MM_F_SOFTCLIP;
    }
    if (mapping_options.secondary_seq) {
        m_mapping_options->flag |= MM_F_SECONDARY_SEQ;
    }
    spdlog::trace("> Map parameters input by user: soft clipping={} and secondary seq={}.",
                  static_cast<bool>(m_mapping_options->flag & MM_F_SOFTCLIP),
                  static_cast<bool>(m_mapping_options->flag & MM_F_SECONDARY_SEQ));

    m_mapping_options->occ_dist = mapping_options.occ_dist.value_or(m_mapping_options->occ_dist);
    m_mapping_options->min_chain_score =
            mapping_options.min_chain_score.value_or(m_mapping_options->min_chain_score);
    m_mapping_options->zdrop = mapping_options.zdrop.value_or(m_mapping_options->zdrop);
    m_mapping_options->zdrop_inv = mapping_options.zdrop_inv.value_or(m_mapping_options->zdrop_inv);
    m_mapping_options->mini_batch_size =
            mapping_options.mini_batch_size.value_or(m_mapping_options->mini_batch_size);
    // Force cigar generation.
    m_mapping_options->flag |= MM_F_CIGAR;
    if (mapping_options.cs) {
        m_mapping_options->flag |= MM_F_OUT_CS | MM_F_CIGAR;
        if (*mapping_options.cs == "short") {
            m_mapping_options->flag &= ~MM_F_OUT_CS_LONG;
        } else if (*mapping_options.cs == "long") {
            m_mapping_options->flag |= MM_F_OUT_CS_LONG;
        } else if (*mapping_options.cs == "none") {
            m_mapping_options->flag &= ~MM_F_OUT_CS;
        } else {
            spdlog::warn("Unrecognized options for --cs={}", *mapping_options.cs);
        }
    }
    if (mapping_options.dual) {
        if (*mapping_options.dual == "yes") {
            m_mapping_options->flag &= ~MM_F_NO_DUAL;
        } else if (*mapping_options.dual == "no") {
            m_mapping_options->flag |= MM_F_NO_DUAL;
        } else {
            spdlog::warn("Unrecognized options for --dual={}", *mapping_options.dual);
        }
    }

    // Either use the default value for cap_kalloc and max_sw_mat defined in the dorado
    // options initialization, or if it's set to nullopt use the minimap2 library default.
    m_mapping_options->cap_kalloc =
            mapping_options.cap_kalloc.value_or(m_mapping_options->cap_kalloc);
    m_mapping_options->max_sw_mat =
            mapping_options.max_sw_mat.value_or(m_mapping_options->max_sw_mat);
}

std::shared_ptr<mm_idx_t> Minimap2Index::load_initial_index(const std::string& index_file,
                                                            int num_threads,
                                                            bool allow_split_index) {
    m_index_reader = create_index_reader(index_file, *m_index_options);
    std::shared_ptr<mm_idx_t> index(mm_idx_reader_read(m_index_reader.get(), num_threads),
                                    IndexDeleter());
    if (!allow_split_index) {
        // If split index is not supported, then verify that the index doesn't
        // have multiple parts by loading the index again and making sure
        // the returned value is nullptr.
        IndexUniquePtr split_index{};
        split_index.reset(mm_idx_reader_read(m_index_reader.get(), num_threads));
        if (split_index != nullptr) {
            return nullptr;
        }
    }

    if (index->k != m_index_options->k || index->w != m_index_options->w) {
        spdlog::warn(
                "Indexing parameters mismatch prebuilt index: using paramateres kmer "
                "size={} and window size={} from prebuilt index.",
                index->k, index->w);
    }

    if (mm_verbose >= 3) {
        mm_idx_stat(index.get());
    }

    spdlog::debug("Loaded index with {} target seqs", index->n_seq);

    return index;
}

IndexLoadResult Minimap2Index::load_next_chunk(int num_threads) {
    if (!m_index_reader) {
        return IndexLoadResult::no_index_loaded;
    }

    auto next_idx = mm_idx_reader_read(m_index_reader.get(), num_threads);
    if (next_idx == nullptr) {
        return IndexLoadResult::end_of_index;
    }

    m_index.reset(next_idx, IndexDeleter());
    mm_mapopt_update(&m_mapping_options.value(), m_index.get());
    spdlog::debug("Loaded next index chunk with {} target seqs", m_index->n_seq);
    return IndexLoadResult::success;
}

bool Minimap2Index::initialise(Minimap2Options options) {
    m_index_options = std::make_optional<mm_idxopt_t>();
    m_mapping_options = std::make_optional<mm_mapopt_t>();

    mm_set_opt(0, &m_index_options.value(), &m_mapping_options.value());
    if (mm_set_opt(options.mm2_preset.c_str(), &m_index_options.value(),
                   &m_mapping_options.value()) != 0) {
        spdlog::error("Cannot set mm2 options with preset: {}", options.mm2_preset);
        return false;
    }

    set_index_options(options);
    set_mapping_options(options);

    if (mm_check_opt(&m_index_options.value(), &m_mapping_options.value()) < 0) {
        m_index_options = {};
        m_mapping_options = {};
        return false;
    }

    if (options.print_aln_seq) {
        mm_dbg_flag |= MM_DBG_PRINT_QNAME | MM_DBG_PRINT_ALN_SEQ;
    }
    spdlog::debug("> Map parameters input by user: dbg print qname={} and aln seq={}.",
                  static_cast<bool>(mm_dbg_flag & MM_DBG_PRINT_QNAME),
                  static_cast<bool>(mm_dbg_flag & MM_DBG_PRINT_ALN_SEQ));

    m_options = std::move(options);
    return true;
}

IndexLoadResult Minimap2Index::load(const std::string& index_file,
                                    int num_threads,
                                    bool allow_split_index) {
    assert(m_index_options && m_mapping_options &&
           "Loading an index requires options have been initialised.");
    assert(!m_index && "Loading an index requires it is not already loaded.");

    // Check if reference file exists.
    if (!std::filesystem::exists(index_file)) {
        return IndexLoadResult::reference_file_not_found;
    }

    auto index = load_initial_index(index_file, num_threads, allow_split_index);
    if (!index) {
        return IndexLoadResult::split_index_not_supported;
    }

    mm_mapopt_update(&m_mapping_options.value(), index.get());

    if (!m_options.junc_bed.empty()) {
        mm_idx_bed_read(index.get(), m_options.junc_bed.c_str(), 1);
    }

    m_index = index;

    return IndexLoadResult::success;
}

std::shared_ptr<Minimap2Index> Minimap2Index::create_compatible_index(
        const Minimap2Options& options) const {
    assert(static_cast<const Minimap2IndexOptions&>(m_options) == options &&
           " create_compatible_index expects compatible indexing options");
    assert(m_index && " create_compatible_index expects the index has been loaded.");

    auto compatible = std::make_shared<Minimap2Index>();
    if (!compatible->initialise(options)) {
        return {};
    }
    compatible->m_index = m_index;
    mm_mapopt_update(&compatible->m_mapping_options.value(), m_index.get());

    return compatible;
}

HeaderSequenceRecords Minimap2Index::get_sequence_records_for_header() const {
    std::vector<std::pair<char*, uint32_t>> records;
    for (uint32_t i = 0; i < m_index->n_seq; ++i) {
        records.push_back(std::make_pair(m_index->seq[i].name, m_index->seq[i].len));
    }
    return records;
}

const mm_idxopt_t& Minimap2Index::index_options() const {
    assert(m_index_options && "Access to indexing options require they are initialised.");
    return *m_index_options;
}

const mm_mapopt_t& Minimap2Index::mapping_options() const {
    assert(m_mapping_options && "Access to mapping options require they are initialised.");
    return *m_mapping_options;
}

const Minimap2Options& Minimap2Index::get_options() const { return m_options; }

}  // namespace dorado::alignment
