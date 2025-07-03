#pragma once

#include "hts_utils/hts_types.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>

namespace dorado {

namespace hts_writer {

namespace fs = std::filesystem;

enum class StructureMode {

};

class IStructure {
public:
    virtual ~IStructure() = default;
    virtual void init() = 0;
    virtual std::shared_ptr<const fs::path> get_path(const HtsData &item) = 0;
};

class SingleFileStructure : public IStructure {
public:
    SingleFileStructure(fs::path path) : m_path(std::make_shared<fs::path>(std::move(path))) {};
    void init() override { try_create_parent_folder(); }
    std::shared_ptr<const fs::path> get_path(const HtsData &_) override { return m_path; };

private:
    const std::shared_ptr<const fs::path> m_path;

    bool try_create_parent_folder() const;
};

}  // namespace hts_writer
}  // namespace dorado