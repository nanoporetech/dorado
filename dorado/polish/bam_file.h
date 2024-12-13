#pragma once

#include <filesystem>
#include <iosfwd>
#include <memory>
#include <tuple>
#include <vector>

struct htsFile;
struct hts_idx_t;
struct sam_hdr_t;
struct bam1_t;

#ifdef __cplusplus
extern "C" {
#endif
int hts_close(htsFile* fp);
void hts_idx_destroy(hts_idx_t* idx);
void sam_hdr_destroy(sam_hdr_t* bh);
void bam_destroy1(bam1_t* b);
#ifdef __cplusplus
}
#endif

struct HeaderLineData {
    std::string header_type;
    std::vector<std::pair<std::string, std::string>> tags;
};

class BamFile {
public:
    BamFile(const std::filesystem::path& in_fn);

    // Getters.
    htsFile* fp() const { return m_fp.get(); }
    hts_idx_t* idx() const { return m_idx.get(); }
    sam_hdr_t* hdr() const { return m_hdr.get(); }

    htsFile* fp() { return m_fp.get(); }
    hts_idx_t* idx() { return m_idx.get(); }
    sam_hdr_t* hdr() { return m_hdr.get(); }

    std::vector<HeaderLineData> parse_header() const;

    std::unique_ptr<bam1_t, decltype(&bam_destroy1)> get_next();

private:
    std::unique_ptr<htsFile, decltype(&hts_close)> m_fp;
    std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)> m_idx;
    std::unique_ptr<sam_hdr_t, decltype(&sam_hdr_destroy)> m_hdr;
};

void header_to_stream(std::ostream& os, const std::vector<HeaderLineData>& header);

std::string header_to_string(const std::vector<HeaderLineData>& header);
