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

class HtsFileWriter;

class HtsFileWriterBuilder {
public:
    HtsFileWriterBuilder();
    HtsFileWriterBuilder(bool emit_fastq,
                         bool emit_sam,
                         bool reference_requested,
                         const std::optional<std::string>& output_dir,
                         const int writer_threads,
                         const utils::ProgressCallback& progress_callback,
                         const utils::DescriptionCallback& description_callback);

    void set_output_mode(OutputMode output_mode);
    OutputMode get_output_mode() const { return m_output_mode; }

    void set_output_dir(const std::optional<std::string>& output_dir) { m_output_dir = output_dir; }
    std::optional<std::string> get_output_dir() const { return m_output_dir; }

    bool get_sort() const { return m_sort; }

    void set_writer_threads(int threads) { m_writer_threads = threads; }
    int get_writer_threads() const { return m_writer_threads; }

    void set_progress_callback(const utils::ProgressCallback& callback) {
        m_progress_callback = callback;
    }
    void set_description_callback(const utils::DescriptionCallback& callback) {
        m_description_callback = callback;
    }

    void set_is_fd_tty(bool is_fd_tty) { m_is_fd_tty = is_fd_tty; }
    void set_is_fd_pipe(bool is_fd_pipe) { m_is_fd_pipe = is_fd_pipe; }

    void update();
    std::unique_ptr<HtsFileWriter> build();

private:
    bool m_emit_fastq{false}, m_emit_sam{false}, m_reference_requested{false};
    std::optional<std::string> m_output_dir{std::nullopt};
    int m_writer_threads{0};
    utils::ProgressCallback m_progress_callback;
    utils::DescriptionCallback m_description_callback;

    bool m_sort{false};

    OutputMode m_output_mode{OutputMode::BAM};
    bool m_is_fd_tty{false}, m_is_fd_pipe{false};
};

class HtsFileWriter : public IWriter {
public:
    explicit HtsFileWriter(OutputMode mode,
                           int threads,
                           const utils::ProgressCallback& progress_callback,
                           const utils::DescriptionCallback& description_callback)
            : m_mode(mode),
              m_threads(threads),
              m_progress_callback(progress_callback),
              m_description_callback(description_callback) {}
    void take_header(SamHdrPtr header) { m_header = std::move(header); };
    SamHdrPtr& header() { return m_header; }
    void process(const Processable item);

    OutputMode mode() const { return m_mode; }
    virtual bool finalise_is_noop() const = 0;

    int get_threads() const { return m_threads; }

    void set_progress(size_t progress) const { m_progress_callback(progress); }
    void set_description(const std::string& description) const {
        m_description_callback(description);
    }

protected:
    const OutputMode m_mode;
    const int m_threads;
    const utils::ProgressCallback& m_progress_callback;
    const utils::DescriptionCallback& m_description_callback;
    SamHdrPtr m_header{nullptr};

    virtual void handle(const HtsData& _) = 0;
};

class StreamHtsFileWriter : public HtsFileWriter {
public:
    StreamHtsFileWriter(OutputMode mode,
                        int threads,
                        const utils::ProgressCallback& progress_callback,
                        const utils::DescriptionCallback& description_callback);
    void init() override;
    void shutdown() override;

    bool finalise_is_noop() const override { return true; };

private:
    std::unique_ptr<utils::HtsFile> m_hts_file;

    void handle(const HtsData& data) override;
};

class StructuredHtsFileWriter : public HtsFileWriter {
public:
    StructuredHtsFileWriter(const std::string& output_dir,
                            OutputMode mode,
                            int threads,
                            bool sort,
                            const utils::ProgressCallback& progress_callback,
                            const utils::DescriptionCallback& description_callback);
    void init() override;
    void shutdown() override;

    bool finalise_is_noop() const override { return m_mode == OutputMode::FASTQ || !m_sort; };

private:
    const std::string m_output_dir;
    const bool m_sort;
    std::unordered_map<std::string, utils::HtsFile> m_hts_files;

    bool try_create_output_folder() const;

    void handle(const HtsData& data) override;
};

}  // namespace hts_writer

std::string to_string(hts_writer::OutputMode mode);

}  // namespace dorado
