#pragma once

#include "bed_file.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace dorado::alignment {

class BedFileAccess {
    mutable std::mutex m_mutex{};
    std::map<std::string, std::shared_ptr<BedFile>> m_bedfile_lut;

public:
    bool load_bedfile(const std::string& bedfile);

    // Returns the bed-file if already loaded. Empty pointer otherwise.
    std::shared_ptr<BedFile> get_bedfile(const std::string& bedfile);

    // Remove a bedfile entry.
    void remove_bedfile(const std::string& bedfile);
};

}  // namespace dorado::alignment
