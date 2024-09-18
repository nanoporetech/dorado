#pragma once

#include <istream>
#include <map>
#include <string>
#include <vector>

namespace dorado::alignment {

class BedFile {
public:
    struct Entry {
        std::string bed_line{};
        size_t start{};
        size_t end{};
        char strand{'.'};
    };

    using Entries = std::vector<Entry>;

    BedFile() = default;
    BedFile(BedFile&& other) = delete;
    BedFile(const BedFile&) = delete;
    BedFile& operator=(const BedFile&) = delete;
    ~BedFile() = default;

    bool load(const std::string& filename);
    bool load(std::istream& input);

    const Entries& entries(const std::string& genome) const;

    const std::string& filename() const;

private:
    std::map<std::string, Entries> m_genomes;
    std::string m_file_name{"<stream>"};
    static const Entries NO_ENTRIES;
};

bool operator==(const BedFile::Entry& l, const BedFile::Entry& r);

bool operator!=(const BedFile::Entry& l, const BedFile::Entry& r);

}  // namespace dorado::alignment
