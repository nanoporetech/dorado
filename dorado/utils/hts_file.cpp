#include "hts_file.h"

#include <htslib/bgzf.h>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <exception>
#include <filesystem>
#include <map>

namespace {

uint64_t calculate_sorting_key(bam1_t* const record) {
    return (static_cast<uint64_t>(record->core.tid) << 32) | record->core.pos;
}

}  // namespace

namespace dorado::utils {

HtsFile::HtsFile(const std::string& filename, OutputMode mode, size_t threads) {
    switch (mode) {
    case OutputMode::FASTQ:
        m_file = hts_open(filename.c_str(), "wf");
        break;
    case OutputMode::BAM: {
        auto file = filename;
        if (file != "-") {
            file += ".temp";
        }
        m_file = hts_open(file.c_str(), "wb");
    } break;
    case OutputMode::SAM:
        m_file = hts_open(filename.c_str(), "w");
        break;
    case OutputMode::UBAM:
        m_file = hts_open(filename.c_str(), "wb0");
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
    auto temp_filename = std::string(m_file->fn);
    bool is_bgzf = m_file->format.compression == bgzf;

    sam_hdr_destroy(m_header);
    hts_close(m_file);

    if (temp_filename == std::string("-") || !is_bgzf) {
        return;
    }

    std::filesystem::path filepath(temp_filename);
    filepath.replace_extension("");
    auto in_file = hts_open(temp_filename.c_str(), "rb");
    auto out_file = hts_open(filepath.string().c_str(), "wb");

    auto in_header = sam_hdr_read(in_file);
    auto out_header = sam_hdr_dup(in_header);
    sam_hdr_change_HD(out_header, "SO", "coordinate");
    if (sam_hdr_write(out_file, out_header) < 0) {
        spdlog::error("Failed to write header for sorted bam file {}", out_file->fn);
        return;
    }

    auto record = bam_init1();

    std::multimap<uint64_t, int64_t> record_map;
    auto pos = bgzf_tell(in_file->fp.bgzf);
    while ((sam_read1(in_file, in_header, record) >= 0)) {
        auto sorting_key = calculate_sorting_key(record);
        record_map.insert({sorting_key, pos});
        pos = bgzf_tell(in_file->fp.bgzf);
    }

    for (auto [sorting_key, record_offset] : record_map) {
        if (bgzf_seek(in_file->fp.bgzf, record_offset, SEEK_SET) < 0) {
            spdlog::error("Failed to seek in file {}, record offset {}", in_file->fn,
                          record_offset);
            return;
        }
        if (sam_read1(in_file, in_header, record) < 0) {
            spdlog::error("Failed to read from temporary file {}", in_file->fn);
            return;
        }
        if (sam_write1(out_file, out_header, record) < 0) {
            spdlog::error("Failed to write to sorted file {}", out_file->fn);
            return;
        }
    }

    bam_destroy1(record);
    sam_hdr_destroy(in_header);
    sam_hdr_destroy(out_header);
    hts_close(out_file);
    hts_close(in_file);

    if (sam_index_build(filepath.string().c_str(), 0) < 0) {
        spdlog::error("Failed to build index for file {}", out_file->fn);
        return;
    }

    std::filesystem::remove(temp_filename);
}

int HtsFile::set_and_write_header(const sam_hdr_t* const header) {
    if (header) {
        // Avoid leaking memory if this is called twice.
        if (m_header) {
            sam_hdr_destroy(m_header);
        }
        m_header = sam_hdr_dup(header);
        return sam_hdr_write(m_file, m_header);
    }
    return 0;
}

int HtsFile::write(bam1_t* const record) {
    // FIXME -- HtsFile is constructed in a state where attempting to write
    // will segfault, since set_and_write_header has to have been called
    // in order to set m_header.
    assert(m_header);
    return sam_write1(m_file, m_header, record);
}

}  // namespace dorado::utils
