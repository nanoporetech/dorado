#include "medaka_bamiter.h"

#include <errno.h>
#include <string.h>

// iterator for reading bam
int read_bam(void *data, bam1_t *b) {
    mplp_data *aux = (mplp_data *)data;
    uint8_t *tag;
    bool check_tag = (strcmp(aux->tag_name, "") != 0);
    bool have_rg = (aux->read_group != NULL);
    uint8_t *rg;
    char *rg_val;
    int ret;
    while (1) {
        ret = aux->iter ? sam_itr_next(aux->fp, aux->iter, b) : sam_read1(aux->fp, aux->hdr, b);
        if (ret < 0) {
            break;
        }
        // only take primary alignments
        if (b->core.flag &
            (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FQCFAIL | BAM_FDUP)) {
            continue;
        }
        // filter by mapping quality
        if ((int)b->core.qual < aux->min_mapQ) {
            continue;
        }
        // filter by tag
        if (check_tag) {
            tag = bam_aux_get((const bam1_t *)b, aux->tag_name);
            if (tag == NULL) {  // tag isn't present or is currupt
                if (aux->keep_missing) {
                    break;
                } else {
                    continue;
                }
            }
            int tag_value = static_cast<int32_t>(bam_aux2i(tag));
            if (errno == EINVAL) {
                continue;  // tag was not integer
            }
            if (tag_value != aux->tag_value) {
                continue;
            }
        }
        // filter by RG (read group):
        if (have_rg) {
            rg = bam_aux_get((const bam1_t *)b, "RG");
            if (rg == NULL) {
                continue;  // missing
            }
            rg_val = bam_aux2Z(rg);
            if (errno == EINVAL) {
                continue;  // bad parse
            }
            if (strcmp(aux->read_group, rg_val) != 0) {
                continue;  // not wanted
            }
        }
        break;
    }
    return ret;
}

BamFile::BamFile(const std::filesystem::path &in_fn)
        : m_fp{hts_open(in_fn.c_str(), "rb"), hts_close},
          m_idx{nullptr, hts_idx_destroy},
          m_hdr{nullptr, sam_hdr_destroy} {
    if (!m_fp) {
        throw std::runtime_error{"Could not open BAM file: '" + in_fn.string() + "'!"};
    }

    m_idx = std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)>(
            sam_index_load(m_fp.get(), in_fn.c_str()), hts_idx_destroy);

    if (!m_idx) {
        throw std::runtime_error{"Could not open index for BAM file: '" + in_fn.string() + "'!"};
    }

    m_hdr = std::unique_ptr<sam_hdr_t, decltype(&sam_hdr_destroy)>(sam_hdr_read(m_fp.get()),
                                                                   sam_hdr_destroy);

    if (!m_hdr) {
        throw std::runtime_error{"Could not load header from BAM file: '" + in_fn.string() + "'!"};
    }

    // Create a unique_ptr with a custom deleter

    // m_fp = std::unique_ptr<htsFile, decltype(&hts_close)>(
    //     hts_open(in_fn.c_str(), "rb"), hts_close
    // );

    // htsFile *m_fp = hts_open(in_fn.c_str(), "rb");

    // m_fp = hts_open(in_fn.c_str(), "rb");
    // m_idx = sam_index_load(m_fp, in_fn.c_str());
    // m_hdr = sam_hdr_read(m_fp);
    // if ((m_hdr == nullptr) || (m_idx == nullptr) || (m_fp == nullptr)) {
    //     throw std::runtime_error
    //     destroy_bam_fset(fset);
    //     fprintf(stderr, "Failed to read .bam file '%s'.", fname);
    //     exit(1);
    // }
}

// BamFile::~BamFile() {
//     hts_close(m_fp);
//     hts_idx_destroy(m_idx);
//     sam_hdr_destroy(m_hdr);
// }
