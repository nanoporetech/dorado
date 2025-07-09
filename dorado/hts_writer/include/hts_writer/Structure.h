#pragma once

#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriter.h"

namespace dorado {

namespace hts_writer {

class IStructure {
public:
    virtual ~IStructure() = default;
    virtual const std::string &get_path(const HtsData &item) = 0;
};

class SingleFileStructure : public IStructure {
public:
    SingleFileStructure(const std::string &output_dir, OutputMode mode);
    const std::string &get_path([[maybe_unused]] const HtsData &hts_data) override;

private:
    const OutputMode m_mode;
    const std::string m_path;

    std::string get_filename() const;

    void create_output_folder() const;
};

}  // namespace hts_writer
}  // namespace dorado