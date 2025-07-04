#pragma once

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriter.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>

namespace dorado {

namespace hts_writer {

namespace fs = std::filesystem;

class IStructure {
public:
    virtual ~IStructure() = default;
    virtual void init() = 0;
    virtual std::shared_ptr<const fs::path> get_path(const HtsData &item) = 0;
};

class SingleFileStructure : public IStructure {
public:
    SingleFileStructure(const fs::path &output_dir, OutputMode mode)
            : m_mode(mode), m_path(make_shared_path(output_dir)) {};
    void init() override { try_create_output_folder(); }
    std::shared_ptr<const fs::path> get_path([[maybe_unused]] const HtsData &_) override {
        return m_path;
    };

private:
    const OutputMode m_mode;
    const std::shared_ptr<const fs::path> m_path;

    bool try_create_output_folder() const;

    std::shared_ptr<const fs::path> make_shared_path(const fs::path &output_dir) const;
    std::string get_filename() const;
};

}  // namespace hts_writer
}  // namespace dorado