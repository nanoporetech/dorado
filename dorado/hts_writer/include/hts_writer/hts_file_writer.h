#pragma once

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "interface.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace dorado {

namespace hts_writer {

namespace ext {
constexpr std::string_view BAM{".bam"};
constexpr std::string_view SAM{".sam"};
constexpr std::string_view FASTQ{".fastq"};
}  // namespace ext

using OutputMode = utils::HtsFile::OutputMode;

class HtsFileWriter : public IWriter {
public:
    explicit HtsFileWriter(OutputMode mode,
                           int threads,
                           utils::ProgressCallback progress_callback,
                           utils::DescriptionCallback description_callback)
            : m_mode(mode),
              m_threads(threads),
              m_progress_callback(std::move(progress_callback)),
              m_description_callback(std::move(description_callback)) {}
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
    utils::ProgressCallback m_progress_callback;
    utils::DescriptionCallback m_description_callback;
    SamHdrPtr m_header{nullptr};

    virtual void handle(const HtsData& _) = 0;
};

class SingleHtsFileWriter : public HtsFileWriter {
public:
    SingleHtsFileWriter(OutputMode mode,
                        int threads,
                        utils::ProgressCallback progress_callback,
                        utils::DescriptionCallback description_callback);
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
                            utils::ProgressCallback progress_callback,
                            utils::DescriptionCallback description_callback);
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

}  // namespace dorado
