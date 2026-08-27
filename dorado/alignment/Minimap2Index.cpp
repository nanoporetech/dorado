#include "alignment/Minimap2Index.h"

#include "alignment/minimap2_wrappers.h"
#include "hts_utils/FastxSequentialReader.h"
#include "hts_utils/header_sq_record.h"
#include "hts_utils/hts_types.h"
#include "hts_utils/sequence_file_format.h"

#include <spdlog/spdlog.h>

//todo: mmpriv.h is a private header from mm2 for the mm_event_identity function.
//Ask lh3 t  make some of these funcs publicly available?
#include <mmpriv.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

struct IndexDeleter {
    void operator()(mm_idx_t* index) { mm_idx_destroy(index); }
};
using IndexUniquePtr = std::unique_ptr<mm_idx_t, IndexDeleter>;

dorado::alignment::IndexReader create_index_reader(const std::string& index_file,
                                                   const mm_idxopt_t& index_options) {
    dorado::alignment::IndexReaderPtr reader;
    reader.reset(mm_idx_reader_open(index_file.c_str(), &index_options, 0));

    return dorado::alignment::IndexReader{std::move(reader), index_file};
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
    m_index_reader = create_index_reader(index_file, m_options.index.index_options->get());
    if (!m_index_reader.inner) {
        // Reason could be not having permissions to open the file
        return {nullptr, IndexLoadResult::file_open_error};
    }
    std::shared_ptr<mm_idx_t> index(mm_idx_reader_read(m_index_reader.inner.get(), num_threads),
                                    IndexDeleter());

    if (!index) {
        m_index_reader.inner.reset();
        return {nullptr, IndexLoadResult::end_of_index};
    }

    if (index->n_seq == 0) {
        // An empty or improperly formatted file can result in an mm_idx_t object with
        // no indexes in it. Check for this, and treat it as a load failure.
        m_index_reader.inner.reset();
        return {nullptr, IndexLoadResult::end_of_index};
    }

    if (index->k != m_options.index.index_options->get().k ||
        index->w != m_options.index.index_options->get().w) {
        spdlog::warn(
                "Indexing parameters mismatch prebuilt index: using parameters kmer "
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
    if (!m_index_reader.inner) {
        if (m_indexes.empty()) {
            return IndexLoadResult::no_index_loaded;
        }
        return IndexLoadResult::end_of_index;
    }

    std::shared_ptr<const mm_idx_t> next_idx(
            mm_idx_reader_read(m_index_reader.inner.get(), num_threads), IndexDeleter());
    if (!next_idx) {
        m_index_reader.inner.reset();
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
    if (mm_check_opt(&options.index.index_options->get(), &options.mapping.mapping_options->get()) <
        0) {
        return false;
    }

    m_options = std::move(options);
    return true;
}

IndexLoadResult Minimap2Index::load(const std::string& index_file,
                                    int num_threads,
                                    bool incremental_load) {
    assert(m_options.index.index_options && m_options.mapping.mapping_options &&
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

    if (!m_options.index.junc_bed.empty()) {
        mm_idx_bed_read(index.get(), m_options.index.junc_bed.c_str(), 1);
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
    assert(m_options.index == options.index &&
           " create_compatible_index expects compatible indexing options");
    assert(!m_indexes.empty() && " create_compatible_index expects the index has been loaded.");

    auto compatible = std::make_shared<Minimap2Index>(m_skip_header_cache);
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
    m_header_records_cache.clear();
    mm_mapopt_update(&m_options.mapping.mapping_options->get(), index.get());
    cache_header_records(*index);
    m_indexes.emplace_back(std::move(index));
}

void Minimap2Index::add_index(std::shared_ptr<const mm_idx_t> index) {
    cache_header_records(*index);
    m_indexes.emplace_back(std::move(index));
}

const utils::HeaderSQRecords& Minimap2Index::get_sequence_records_for_header() const {
    assert(!m_skip_header_cache);
    return m_header_records_cache;
}

void Minimap2Index::cache_header_records(const mm_idx_t& index) {
    if (m_skip_header_cache) {
        return;
    }

    m_header_records_cache.reserve(m_header_records_cache.size() + index.n_seq);
    const std::filesystem::path reference_path = m_index_reader.file;

    // Try to open the FASTA reference extract the reference sequences for encoding
    // into SQ M5 hashes. If the file isn't a FASTA we omit the M5 and UR tags because
    // the FASTA references can be any string (IUPAC+) while minimap stores the lossy
    // seq4 (ACGTN) encoding.
    // Because htslib doesn't support loading mmi indexes and the lossy seq4 encoding,
    // htslib will either crash or error with `SQ header M5 tag discrepancy for reference X`
    {
        const auto ref_fmt = hts_io::parse_sequence_format(reference_path);
        const bool is_fasta = ref_fmt == hts_io::SequenceFormatType::FASTA;

        if (!is_fasta) {
            // Index not supported by htslib (likely mmi or fastq) - do not add SQ M5 or UR tags.
            spdlog::debug("Omitting SQ M5/UR tags as reference is not FASTA '{}'.",
                          reference_path.string());
            for (uint32_t j = 0; j < index.n_seq; ++j) {
                utils::HeaderSQRecord record{std::string(index.seq[j].name), index.seq[j].len};
                m_header_records_cache.emplace_back(std::move(record));
            }
            return;
        }
    }

    // Resolve the path for the UR tag.
    const std::shared_ptr<const std::string> uri = std::make_shared<std::string>(
            "file://" + std::filesystem::weakly_canonical(m_index_reader.file).string());

    // Open the FASTA Reader
    std::unique_ptr<hts_io::FastxSequentialReader> fasta_reader;
    try {
        fasta_reader = std::make_unique<hts_io::FastxSequentialReader>(reference_path);
    } catch (const std::exception& exc) {
        spdlog::error("Failed to open FASTA reference '{}' for SQ M5 calculation: '{}'.",
                      reference_path.string(), exc.what());
        throw;
    }

    spdlog::debug("Computing SQ M5 hashes.");
    std::unordered_set<std::string> sequence_names;
    std::for_each(index.seq, std::next(index.seq, index.n_seq),
                  [&sequence_names](const mm_idx_seq_t& idx_seq) {
                      sequence_names.insert(idx_seq.name);
                  });

    std::unordered_map<std::string, std::pair<utils::MD5Hex, uint32_t>> reference_info;
    {
        utils::MD5Generator md5gen;
        hts_io::FastxRecord fastx_record;
        while (reference_info.size() != sequence_names.size() &&
               fasta_reader->get_next(fastx_record)) {
            std::string ref_name = std::string(fastx_record.name);
            if (!sequence_names.contains(ref_name)) {
                continue;
            }
            auto& [md5, len] = reference_info[ref_name];
            md5gen.get_sequence_md5(md5, fastx_record.seq);
            len = static_cast<uint32_t>(fastx_record.seq.length());
        }
    }

    // Lookup each sequence by name and compute the MD5 hash.
    for (uint32_t j = 0; j < index.n_seq; ++j) {
        std::string ref_sequence_name = index.seq[j].name;
        const uint32_t ref_sequence_length = index.seq[j].len;

        auto info = reference_info.find(ref_sequence_name);
        if (info == std::end(reference_info)) {
            throw std::runtime_error(fmt::format("Reference sequence '{}' not found in '{}'.",
                                                 ref_sequence_name, reference_path.string()));
        }

        if (info->second.second != ref_sequence_length) {
            throw std::runtime_error(fmt::format(
                    "Reference sequence '{}' length mismatch for '{}': expected {}, got {}.",
                    ref_sequence_name, reference_path.string(), ref_sequence_length,
                    info->second.second));
        }

        utils::HeaderSQRecord record{
                .sequence_name = std::move(ref_sequence_name),
                .length = ref_sequence_length,
                .uri = uri,
        };

        std::strncpy(record.md5, info->second.first, std::size(info->second.first));
        m_header_records_cache.emplace_back(std::move(record));
    }

    spdlog::debug("Finished computing SQ M5 hashes.");
}

const mm_idxopt_t& Minimap2Index::index_options() const {
    assert(m_options.index.index_options &&
           "Access to indexing options require they are initialised.");
    return m_options.index.index_options->get();
}

const mm_mapopt_t& Minimap2Index::mapping_options() const {
    assert(m_options.mapping.mapping_options &&
           "Access to mapping options require they are initialised.");
    return m_options.mapping.mapping_options->get();
}

const Minimap2Options& Minimap2Index::get_options() const { return m_options; }

}  // namespace dorado::alignment
