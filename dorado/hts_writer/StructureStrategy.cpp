#include "hts_writer/StructureStrategy.h"

namespace dorado {

namespace hts_writer {

bool SingleFileStructure::try_create_parent_folder() const {
    const auto parent = m_path->parent_path();
    std::error_code creation_error;
    // N.B. No error code if folder already exists.
    fs::create_directories(parent, creation_error);
    if (creation_error) {
        spdlog::error("Unable to create output folder '{}'.  ErrorCode({}) {}", parent.string(),
                      creation_error.value(), creation_error.message());
        return false;
    }
    return true;
}

}  // namespace hts_writer
}  // namespace dorado