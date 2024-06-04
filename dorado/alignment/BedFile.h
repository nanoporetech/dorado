#pragma once

#include <map>
#include <string>
#include <vector>

namespace dorado::alignment {

class BedFile {
public:
    struct Entry {
        std::string bed_line;
        size_t start;
        size_t end;
        char strand;
    };

    using Entries = std::vector<Entry>;

private:
    std::map<std::string, Entries> m_genomes;
    std::string m_file_name{};
    static const Entries NO_ENTRIES;

public:
    BedFile() = default;
    BedFile(BedFile&& other) = default;
    BedFile(const BedFile&) = default;
    BedFile& operator=(const BedFile&) = default;
    ~BedFile() = default;

    bool load(const std::string& filename);

    const Entries& entries(const std::string& genome) const;

    const std::string& filename() const;
};

}  // namespace dorado::alignment
