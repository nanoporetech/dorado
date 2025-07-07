#pragma once

#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriter.h"

#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>

namespace dorado {

namespace hts_writer {

class IStructure {
public:
    virtual ~IStructure() = default;
    virtual void init() = 0;
    virtual std::shared_ptr<const std::string> get_path(const HtsData &item) = 0;
};

class SingleFileStructure : public IStructure {
public:
    SingleFileStructure(const std::string &output_dir, OutputMode mode)
            : m_mode(mode), m_path(make_shared_path(output_dir)) {};
    void init() override { create_output_folder(); }
    std::shared_ptr<const std::string> get_path([[maybe_unused]] const HtsData &_) override {
        return m_path;
    };

private:
    const OutputMode m_mode;
    const std::shared_ptr<const std::string> m_path;

    void create_output_folder() const;

    std::shared_ptr<const std::string> make_shared_path(const std::string &output_dir) const;
    std::string get_filename() const;
};

}  // namespace hts_writer
}  // namespace dorado