#include "Minimap2Index.h"

#include "minimap2_wrappers.h"

#include <spdlog/spdlog.h>

//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include <mmpriv.h>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

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

const mm_idx_t* Minimap2Index::index() const {
    if (m_indexes.size() != 1) {
        throw std::range_error("Minimap2Index::index can only be called for non-split indexes.");
    }
    return m_indexes[0].get();
}

const mm_idx_t* Minimap2Index::index(size_t n) const {
    if (n >= m_indexes.size()) {
        throw std::range_error("Invalid index block requested.");
    }
    return m_indexes[n].get();
}

std::pair<std::shared_ptr<mm_idx_t>, IndexLoadResult> Minimap2Index::load_initial_index(
        const std::string& index_file,
        int num_threads) {
    m_index_reader = create_index_reader(index_file, m_options.index_options->get());
    if (!m_index_reader) {
        // Reason could be not having permissions to open the file
        return {nullptr, IndexLoadResult::file_open_error};
    }
    std::shared_ptr<mm_idx_t> index(mm_idx_reader_read(m_index_reader.get(), num_threads),
                                    IndexDeleter());

    if (!index) {
        m_index_reader.reset();
        return {nullptr, IndexLoadResult::end_of_index};
    }

    if (index->n_seq == 0) {
        // An empty or improperly formatted file can result in an mm_idx_t object with
        // no indexes in it. Check for this, and treat it as a load failure.
        m_index_reader.reset();
        return {nullptr, IndexLoadResult::end_of_index};
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

    return {index, IndexLoadResult::success};
}

IndexLoadResult Minimap2Index::load_next_chunk(int num_threads) {
    if (!m_index_reader) {
        if (m_indexes.empty()) {
            return IndexLoadResult::no_index_loaded;
        }
        return IndexLoadResult::end_of_index;
    }

    std::shared_ptr<const mm_idx_t> next_idx(mm_idx_reader_read(m_index_reader.get(), num_threads),
                                             IndexDeleter());
    if (!next_idx) {
        m_index_reader.reset();
        return IndexLoadResult::end_of_index;
    }

    if (m_incremental_load) {
        set_index(std::move(next_idx));
    } else {
        add_index(std::move(next_idx));
    }

    spdlog::debug("Loaded next index chunk with {} target seqs", m_indexes.back()->n_seq);
    return IndexLoadResult::success;
}

bool Minimap2Index::initialise(Minimap2Options options) {
    if (mm_check_opt(&options.index_options->get(), &options.mapping_options->get()) < 0) {
        return false;
    }

    m_options = std::move(options);
    return true;
}

IndexLoadResult Minimap2Index::load(const std::string& index_file,
                                    int num_threads,
                                    bool incremental_load) {
    assert(m_options.index_options && m_options.mapping_options &&
           "Loading an index requires options have been initialised.");
    assert(m_indexes.empty() && "Loading an index requires it is not already loaded.");

    m_incremental_load = incremental_load;

    // Check if reference file exists.
    if (!std::filesystem::exists(index_file)) {
        return IndexLoadResult::reference_file_not_found;
    }

    auto [index, result] = load_initial_index(index_file, num_threads);
    if (result != IndexLoadResult::success) {
        return result;
    }

    if (!m_options.junc_bed.empty()) {
        mm_idx_bed_read(index.get(), m_options.junc_bed.c_str(), 1);
    }

    set_index(std::move(index));
    if (!incremental_load) {
        while (true) {
            result = load_next_chunk(num_threads);
            if (result == IndexLoadResult::end_of_index) {
                break;
            } else if (result != IndexLoadResult::success) {
                return result;
            }
        }
    }

    return IndexLoadResult::success;
}

std::shared_ptr<Minimap2Index> Minimap2Index::create_compatible_index(
        const Minimap2Options& options) const {
    assert(static_cast<const Minimap2IndexOptions&>(m_options) == options &&
           " create_compatible_index expects compatible indexing options");
    assert(!m_indexes.empty() && " create_compatible_index expects the index has been loaded.");

    auto compatible = std::make_shared<Minimap2Index>();
    if (!compatible->initialise(options)) {
        return {};
    }

    compatible->set_index(m_indexes[0]);
    for (size_t i = 1; i < m_indexes.size(); ++i) {
        compatible->add_index(m_indexes[i]);
    }

    return compatible;
}

void Minimap2Index::set_index(std::shared_ptr<const mm_idx_t> index) {
    m_indexes.clear();
    mm_mapopt_update(&m_options.mapping_options->get(), index.get());
    m_indexes.emplace_back(std::move(index));
}

void Minimap2Index::add_index(std::shared_ptr<const mm_idx_t> index) {
    m_indexes.emplace_back(std::move(index));
}

HeaderSequenceRecords Minimap2Index::get_sequence_records_for_header() const {
    HeaderSequenceRecords records;
    for (const auto& index : m_indexes) {
        for (uint32_t j = 0; j < index->n_seq; ++j) {
            records.emplace_back(
                    std::make_pair(std::string(index->seq[j].name), index->seq[j].len));
        }
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
