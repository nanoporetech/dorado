#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct faidx_t;

namespace dorado::hts_io {

struct FaidxDestructor {
    void operator()(faidx_t*);
};

using FaidxPtr = std::unique_ptr<faidx_t, FaidxDestructor>;

// Class to wrap reading randomly from a FASTx
// file using an index via the htslib APIs.
class FastxRandomReader {
    FaidxPtr m_faidx{nullptr};

public:
    FastxRandomReader(const std::filesystem::path& fastx_path);
    ~FastxRandomReader() = default;

    std::string fetch_seq(const std::string& read_id) const;
    std::vector<uint8_t> fetch_qual(const std::string& read_id) const;

    faidx_t* get_raw_faidx_ptr();

    int num_entries() const;
};

}  // namespace dorado::hts_io
