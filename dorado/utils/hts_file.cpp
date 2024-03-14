#include "hts_file.h"

#include <htslib/bgzf.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <map>
#include <stdexcept>

namespace {

uint64_t calculate_sorting_key(bam1_t* const record) {
    return (static_cast<uint64_t>(record->core.tid) << 32) | record->core.pos;
}

}  // namespace

namespace dorado::utils {

HtsFile::HtsFile(const std::string& filename, OutputMode mode, size_t threads) {
    switch (mode) {
    case OutputMode::FASTQ:
        m_file.reset(hts_open(filename.c_str(), "wf"));
        break;
    case OutputMode::BAM: {
        auto file = filename;
        if (file != "-") {
            file += ".temp";
        }
        m_file.reset(hts_open(file.c_str(), "wb"));
    } break;
    case OutputMode::SAM:
        m_file.reset(hts_open(filename.c_str(), "w"));
        break;
    case OutputMode::UBAM:
        m_file.reset(hts_open(filename.c_str(), "wb0"));
        break;
    default:
        throw std::runtime_error("Unknown output mode selected: " +
                                 std::to_string(static_cast<int>(mode)));
    }
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    if (m_file->format.compression == bgzf) {
        auto res = bgzf_mt(m_file->fp.bgzf, int(threads), 128);
        if (res < 0) {
            throw std::runtime_error("Could not enable multi threading for BAM generation.");
        }
    }
}

HtsFile::~HtsFile() {
    if (!m_finalised) {
        spdlog::error("finalise() not called on a HtsFile.");
        // Can't throw in a dtor, and this is a logic_error rather than being data dependent.
        std::abort();
    }
}

// When we close the underlying htsFile, then and only then can we correctly read the virtual file offsets of
// the records (since bgzf_tell doesn't give the correct values in a write-file - see bgzf_flush, etc)
// in order to generate a map of sort coordinates to virtual file offsets. we can then jump around in the
// file to write out the records in the sorted order. finally we can delete the unsorted file.
// in case an error occurs, the unsorted file is left on disk, so users can recover their data.
void HtsFile::finalise() {
    if (std::exchange(m_finalised, true)) {
        spdlog::error("finalise() called twice on a HtsFile. Ignoring second call.");
        return;
    }

    auto temp_filename = std::string(m_file->fn);
    bool is_bgzf = m_file->format.compression == bgzf;

    m_header.reset();
    m_file.reset();

    if (temp_filename == std::string("-") || !is_bgzf) {
        return;
    }

    std::filesystem::path filepath(temp_filename);
    filepath.replace_extension("");

    {
        HtsFilePtr in_file(hts_open(temp_filename.c_str(), "rb"));
        HtsFilePtr out_file(hts_open(filepath.string().c_str(), "wb"));

        SamHdrPtr in_header(sam_hdr_read(in_file.get()));
        SamHdrPtr out_header(sam_hdr_dup(in_header.get()));
        sam_hdr_change_HD(out_header.get(), "SO", "coordinate");
        if (sam_hdr_write(out_file.get(), out_header.get()) < 0) {
            spdlog::error("Failed to write header for sorted bam file {}", out_file->fn);
            return;
        }

        BamPtr record(bam_init1());

        std::multimap<uint64_t, int64_t> record_map;
        auto pos = bgzf_tell(in_file->fp.bgzf);
        while ((sam_read1(in_file.get(), in_header.get(), record.get()) >= 0)) {
            auto sorting_key = calculate_sorting_key(record.get());
            record_map.insert({sorting_key, pos});
            pos = bgzf_tell(in_file->fp.bgzf);
        }

        for (auto [sorting_key, record_offset] : record_map) {
            if (bgzf_seek(in_file->fp.bgzf, record_offset, SEEK_SET) < 0) {
                spdlog::error("Failed to seek in file {}, record offset {}", in_file->fn,
                              record_offset);
                return;
            }
            if (sam_read1(in_file.get(), in_header.get(), record.get()) < 0) {
                spdlog::error("Failed to read from temporary file {}", in_file->fn);
                return;
            }
            if (sam_write1(out_file.get(), out_header.get(), record.get()) < 0) {
                spdlog::error("Failed to write to sorted file {}", out_file->fn);
                return;
            }
        }
    }

    if (sam_index_build(filepath.string().c_str(), 0) < 0) {
        spdlog::error("Failed to build index for file {}", filepath.string());
        return;
    }

    std::filesystem::remove(temp_filename);
}

int HtsFile::set_and_write_header(const sam_hdr_t* const header) {
    if (header) {
        m_header.reset(sam_hdr_dup(header));
        return sam_hdr_write(m_file.get(), m_header.get());
    }
    return 0;
}

int HtsFile::write(const bam1_t* const record) {
    // FIXME -- HtsFile is constructed in a state where attempting to write
    // will segfault, since set_and_write_header has to have been called
    // in order to set m_header.
    assert(m_header);
    return sam_write1(m_file.get(), m_header.get(), record);
}

}  // namespace dorado::utils
