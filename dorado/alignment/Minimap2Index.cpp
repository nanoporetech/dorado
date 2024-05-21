#include "Minimap2Index.h"

#include "minimap2_wrappers.h"

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

std::shared_ptr<mm_idx_t> Minimap2Index::load_initial_index(const std::string& index_file,
                                                            int num_threads,
                                                            bool allow_split_index) {
    m_index_reader = create_index_reader(index_file, m_options.index_options->get());
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

    if (index->k != m_options.index_options->get().k ||
        index->w != m_options.index_options->get().w) {
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

    std::shared_ptr<const mm_idx_t> next_idx(mm_idx_reader_read(m_index_reader.get(), num_threads),
                                             IndexDeleter());
    if (!next_idx) {
        return IndexLoadResult::end_of_index;
    }

    set_index(std::move(next_idx));

    spdlog::debug("Loaded next index chunk with {} target seqs", m_index->n_seq);
    return IndexLoadResult::success;
}

bool Minimap2Index::initialise(Minimap2Options options) {
    if (options.print_aln_seq) {
        mm_dbg_flag |= MM_DBG_PRINT_QNAME | MM_DBG_PRINT_ALN_SEQ;
    }
    spdlog::debug("> Map parameters input by user: dbg print qname={} and aln seq={}.",
                  static_cast<bool>(mm_dbg_flag & MM_DBG_PRINT_QNAME),
                  static_cast<bool>(mm_dbg_flag & MM_DBG_PRINT_ALN_SEQ));

    if (mm_check_opt(&options.index_options->get(), &options.mapping_options->get()) < 0) {
        return false;
    }

    m_options = std::move(options);
    return true;
}

IndexLoadResult Minimap2Index::load(const std::string& index_file,
                                    int num_threads,
                                    bool allow_split_index) {
    assert(m_options.index_options && m_options.mapping_options &&
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

    if (!m_options.junc_bed.empty()) {
        mm_idx_bed_read(index.get(), m_options.junc_bed.c_str(), 1);
    }

    set_index(std::move(index));

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

    compatible->set_index(m_index);

    return compatible;
}

void Minimap2Index::set_index(std::shared_ptr<const mm_idx_t> index) {
    mm_mapopt_update(&m_options.mapping_options->get(), index.get());
    m_index = std::move(index);
}

HeaderSequenceRecords Minimap2Index::get_sequence_records_for_header() const {
    std::vector<std::pair<char*, uint32_t>> records;
    for (uint32_t i = 0; i < m_index->n_seq; ++i) {
        records.push_back(std::make_pair(m_index->seq[i].name, m_index->seq[i].len));
    }
    return records;
}

const mm_idxopt_t& Minimap2Index::index_options() const {
    assert(m_options.index_options && "Access to indexing options require they are initialised.");
    return m_options.index_options->get();
}

const mm_mapopt_t& Minimap2Index::mapping_options() const {
    assert(m_options.mapping_options && "Access to mapping options require they are initialised.");
    return m_options.mapping_options->get();
}

const Minimap2Options& Minimap2Index::get_options() const { return m_options; }

}  // namespace dorado::alignment
