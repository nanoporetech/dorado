#include "BedFileAccess.h"

#include <cassert>

namespace dorado::alignment {

bool BedFileAccess::load_bedfile(const std::string& bedfile) {
    auto bed = std::make_shared<BedFile>();
    if (!bed->load(bedfile)) {
        return false;
    }
    m_bedfile_lut.emplace(bedfile, std::move(bed));
    return true;
}

std::shared_ptr<BedFile> BedFileAccess::get_bedfile(const std::string& bedfile) {
    auto iter = m_bedfile_lut.find(bedfile);
    if (iter == m_bedfile_lut.end()) {
        return {};
    }
    return iter->second;
}

}  // namespace dorado::alignment
