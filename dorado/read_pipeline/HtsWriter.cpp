#include "HtsWriter.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/sequence_utils.h"

#include <htslib/bgzf.h>
#include <htslib/kroundup.h>
#include <htslib/sam.h>
#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <stdexcept>

namespace dorado {

namespace {

uint64_t calculate_sorting_key(bam1_t* const record) {
    return (static_cast<uint64_t>(record->core.tid) << 32) | record->core.pos;
}

}  // namespace

class HtsWriter::HtsFile {
    htsFile* m_file{nullptr};
    sam_hdr_t* m_header{nullptr};

public:
    HtsFile(const std::string& filename, HtsWriter::OutputMode mode, size_t threads);
    ~HtsFile();

    int set_and_write_header(const sam_hdr_t* const header);
    int write(bam1_t* const record);
};

HtsWriter::HtsFile::HtsFile(const std::string& filename,
                            HtsWriter::OutputMode mode,
                            size_t threads) {
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

HtsWriter::HtsFile::~HtsFile() {
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

    std::map<uint64_t, int64_t> record_map;
    auto pos = bgzf_tell(in_file->fp.bgzf);
    while ((sam_read1(in_file, in_header, record) >= 0)) {
        auto sorting_key = calculate_sorting_key(record);
        record_map[sorting_key] = pos;
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

int HtsWriter::HtsFile::set_and_write_header(const sam_hdr_t* const header) {
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

int HtsWriter::HtsFile::write(bam1_t* const record) {
    // FIXME -- HtsFile is constructed in a state where attempting to write
    // will segfault, since set_and_write_header has to have been called
    // in order to set m_header.
    assert(m_header);
    auto res = sam_write1(m_file, m_header, record);
    if (res < 0) {
        throw std::runtime_error("Failed to write SAM record, error code " + std::to_string(res));
    }

    return res;
}

HtsWriter::HtsWriter(const std::string& filename, OutputMode mode, size_t threads)
        : MessageSink(10000, 1), m_file(std::make_unique<HtsFile>(filename, mode, threads)) {
    start_input_processing(&HtsWriter::input_thread_fn, this);
}

HtsWriter::~HtsWriter() { stop_input_processing(); }

HtsWriter::OutputMode HtsWriter::get_output_mode(const std::string& mode) {
    if (mode == "sam") {
        return OutputMode::SAM;
    } else if (mode == "bam") {
        return OutputMode::BAM;
    } else if (mode == "fastq") {
        return OutputMode::FASTQ;
    }
    throw std::runtime_error("Unknown output mode: " + mode);
}

void HtsWriter::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        // If this message isn't a BamPtr, ignore it.
        if (!std::holds_alternative<BamPtr>(message)) {
            continue;
        }

        auto aln = std::move(std::get<BamPtr>(message));
        write(aln.get());

        // For the purpose of estimating write count, we ignore duplex reads
        int64_t dx_tag = 0;
        auto tag_str = bam_aux_get(aln.get(), "dx");
        if (tag_str) {
            dx_tag = bam_aux2i(tag_str);
        }

        bool ignore_read_id = dx_tag == 1;

        if (ignore_read_id) {
            // Read is a duplex read.
            m_duplex_reads_written++;
        } else {
            std::string read_id;

            // If read is a split read, use the parent read id
            // to track write count since we don't know a priori
            // how many split reads will be generated.
            auto pid_tag = bam_aux_get(aln.get(), "pi");
            if (pid_tag) {
                read_id = std::string(bam_aux2Z(pid_tag));
                m_split_reads_written++;
            } else {
                read_id = bam_get_qname(aln.get());
            }

            m_processed_read_ids.insert(std::move(read_id));
        }
    }
}

int HtsWriter::write(bam1_t* const record) {
    // track stats
    m_total++;
    if (record->core.flag & BAM_FUNMAP) {
        m_unmapped++;
    }
    if (record->core.flag & BAM_FSECONDARY) {
        m_secondary++;
    }
    if (record->core.flag & BAM_FSUPPLEMENTARY) {
        m_supplementary++;
    }
    m_primary = m_total - m_secondary - m_supplementary - m_unmapped;

    // Verify that the MN tag, if it exists, and the sequence length are in sync.
    if (auto tag = bam_aux_get(record, "MN"); tag != nullptr) {
        if (bam_aux2i(tag) != record->core.l_qseq) {
            throw std::runtime_error("MN tag and sequence length are not in sync.");
        };
    }

    return m_file->write(record);
}

int HtsWriter::set_and_write_header(const sam_hdr_t* const header) {
    return m_file->set_and_write_header(header);
}

stats::NamedStats HtsWriter::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["unique_simplex_reads_written"] = static_cast<double>(m_processed_read_ids.size());
    stats["duplex_reads_written"] = static_cast<double>(m_duplex_reads_written.load());
    stats["split_reads_written"] = static_cast<double>(m_split_reads_written.load());
    return stats;
}

}  // namespace dorado
