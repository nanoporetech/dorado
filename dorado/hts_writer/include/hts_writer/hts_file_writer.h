#pragma once

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "interface.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace dorado {
namespace hts_writer {

namespace ext {
constexpr std::string_view BAM{".bam"};
constexpr std::string_view SAM{".sam"};
constexpr std::string_view FASTQ{".fastq"};
}  // namespace ext

enum class OutputMode {
    BAM,
    SAM,
    FASTQ,
};

class HtsFileWriter : public IWriter {
public:
    explicit HtsFileWriter(OutputMode mode, int threads, const utils::ProgressCallback& callback)
            : m_mode(mode), m_threads(threads), m_callback(callback) {}
    void take_header(SamHdrPtr header) { m_header = std::move(header); };
    SamHdrPtr& header() { return m_header; }
    void process(const Processable item);

protected:
    const OutputMode m_mode;
    const int m_threads;
    const utils::ProgressCallback& m_callback;
    SamHdrPtr m_header{nullptr};

    virtual void handle(const HtsData& _) = 0;
    virtual void write_header() = 0;

    void update_progress(size_t progress) { m_callback(progress); }
};

class HtsFileWriterBuilder {
public:
    HtsFileWriterBuilder();
    HtsFileWriterBuilder(bool emit_fastq,
                         bool emit_sam,
                         bool reference_requested,
                         const std::optional<std::string>& output_dir,
                         const int writer_threads,
                         utils::ProgressCallback& callback);

    void set_output_from_cli(bool emit_fastq,
                             bool emit_sam,
                             bool reference_requested,
                             const std::optional<std::string>& output_dir);

    void set_output_mode(OutputMode output_mode) { m_output_mode = output_mode; };
    void set_output_dir(const std::optional<std::string>& output_dir) { m_output_dir = output_dir; }
    void set_sort_bam(bool sort_bam) { m_sort_bam = sort_bam; };
    void set_writer_threads(int threads) { m_threads = threads; }
    void set_progress_callback(const utils::ProgressCallback& callback) { m_callback = callback; }

    std::unique_ptr<HtsFileWriter> build() const;

private:
    OutputMode m_output_mode{OutputMode::BAM};
    std::optional<std::string> m_output_dir{std::nullopt};
    bool m_sort_bam{false};
    int m_threads{0};
    utils::ProgressCallback m_callback;
};

std::unique_ptr<HtsFileWriter> build_hts_file_writer(bool emit_fastq,
                                                     bool emit_sam,
                                                     bool reference_requested,
                                                     std::optional<std::string> output_dir,
                                                     const int writer_threads,
                                                     utils::ProgressCallback& callback);

class StreamHtsFileWriter : public HtsFileWriter {
public:
    StreamHtsFileWriter(OutputMode mode, int threads, const utils::ProgressCallback& callback);
    void init() override;
    void shutdown() override;

private:
    std::unique_ptr<utils::HtsFile> m_hts_file;

    void handle(const HtsData& data) override;
    void write_header() override;
};

class StructuredHtsFileWriter : public HtsFileWriter {
public:
    StructuredHtsFileWriter(const std::string& output_dir,
                            OutputMode mode,
                            int threads,
                            const utils::ProgressCallback& callback,
                            bool sort);
    void init() override;
    void shutdown() override;

private:
    const std::string m_output_dir;
    const bool m_sort;
    std::unordered_map<std::string, utils::HtsFile> m_hts_files;

    bool try_create_output_folder() const;

    void handle(const HtsData& data) override;
    void write_header() override;
};

}  // namespace hts_writer
}  // namespace dorado
