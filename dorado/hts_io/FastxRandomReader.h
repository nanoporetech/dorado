#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Class to wrap reading randomly from a FASTx
// file using an index via the htslib APIs.

struct faidx_t;

namespace dorado::hts_io {

struct FaidxDestructor {
    void operator()(faidx_t*);
};

class FastxRandomReader {
    std::unique_ptr<faidx_t, FaidxDestructor> m_faidx;

public:
    FastxRandomReader(const std::filesystem::path& fastx_path);
    ~FastxRandomReader() = default;

    std::string fetch_seq(const std::string& read_id) const;
    std::vector<uint8_t> fetch_qual(const std::string& read_id) const;

    int num_entries() const;
};

}  // namespace dorado::hts_io
