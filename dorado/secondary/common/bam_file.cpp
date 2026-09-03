#include "secondary/common/bam_file.h"

#include "hts_utils/bam_utils.h"

#include <htslib/hts.h>
#include <htslib/sam.h>

#include <stdexcept>

namespace dorado::secondary {

void HtsIdxDestructor::operator()(hts_idx_t* bam) { hts_idx_destroy(bam); }

void HtsItrDestructor::operator()(hts_itr_t* itr) const noexcept { hts_itr_destroy(itr); }

BamIterator::BamIterator(hts_itr_t* itr, htsFile* fp) : m_itr(itr), m_fp(fp) {}

BamPtr BamIterator::get_next() {
    BamPtr rec(bam_init1());

    if (!rec) {
        throw std::runtime_error{"Failed to initialize BAM record"};
    }

    const int32_t ret = sam_itr_next(m_fp, m_itr.get(), rec.get());

    if (ret >= 0) {
        return rec;
    }

    // Iteration finished.
    return nullptr;
}

BamFile::BamFile(const std::filesystem::path& in_fn, int n_threads)
        : m_fp{nullptr}, m_idx{nullptr}, m_hdr{nullptr}, m_n_threads{1} {
    m_fp = HtsFilePtr{hts_open(in_fn.string().c_str(), "rb")};
    if (!m_fp) {
        throw std::runtime_error{"Could not open BAM file: '" + in_fn.string() + "'!"};
    }

    m_idx = HtsIdxPtr{sam_index_load(m_fp.get(), in_fn.string().c_str())};
    if (!m_idx) {
        throw std::runtime_error{
                "Could not open index for BAM file, or the BAM file itself was malformatted: '" +
                in_fn.string() + "'!"};
    }

    m_hdr = SamHdrPtr{sam_hdr_read(m_fp.get())};
    if (!m_hdr) {
        throw std::runtime_error{"Could not load header from BAM file: '" + in_fn.string() + "'!"};
    }

    if (n_threads > 1) {
        hts_set_threads(m_fp.get(), n_threads);
        m_n_threads = n_threads;
    }
}

BamPtr BamFile::get_next() {
    BamPtr record(bam_init1());

    if (record == nullptr) {
        throw std::runtime_error{"Failed to initialize BAM record"};
        return nullptr;
    }

    if (sam_read1(m_fp.get(), m_hdr.get(), record.get()) >= 0) {
        return record;
    }

    return nullptr;
}

BamFileView BamFile::get_view() {
    return BamFileView{.fp = m_fp.get(), .idx = m_idx.get(), .hdr = m_hdr.get()};
}

BamIterator BamFile::fetch(const std::string& chrom, const int64_t start, const int64_t end) {
    const int32_t tid = sam_hdr_name2tid(m_hdr.get(), chrom.c_str());

    if (tid < 0) {
        throw std::runtime_error{"Chromosome not found in BAM header: '" + chrom + "'"};
    }

    hts_itr_t* raw_itr = sam_itr_queryi(m_idx.get(), tid, start, end);

    if (!raw_itr) {
        throw std::runtime_error{"Failed to create iterator for '" + chrom + ":" +
                                 std::to_string(start) + "-" + std::to_string(end) + "'"};
    }

    return BamIterator(raw_itr, m_fp.get());
}

Alignment convert_bam1_to_aln(const bam1_t* b, const sam_hdr_t* hdr) {
    // Compute query start/end by walking the CIGAR.
    int64_t qstart = 0;
    int64_t qspan = 0;
    int64_t rspan = 0;
    std::vector<CigarOp> cigar_vec;

    // Convert the CIGAR and count the alignment length.
    {
        const uint32_t* cigar = bam_get_cigar(b);
        const int32_t n = b->core.n_cigar;

        cigar_vec.resize(n);

        // Find the query start position.
        for (int32_t i = 0; i < n; ++i) {
            const int32_t op = bam_cigar_op(cigar[i]);
            if (op != BAM_CSOFT_CLIP) {
                break;
            }
            const uint32_t len = static_cast<uint32_t>(bam_cigar_oplen(cigar[i]));
            qstart += len;
        }

        for (int32_t i = 0; i < n; ++i) {
            const int32_t op = bam_cigar_op(cigar[i]);
            const uint32_t len = static_cast<uint32_t>(bam_cigar_oplen(cigar[i]));

            cigar_vec[i] = {CIGAR_MM2_TO_DORADO[op], len};

            if ((op == BAM_CSOFT_CLIP) || (op == BAM_CHARD_CLIP)) {
                continue;
            }

            // Consumes query.
            constexpr int32_t CIGAR_OP_CONSUMES_QUERY = 1;
            if (bam_cigar_type(op) & CIGAR_OP_CONSUMES_QUERY) {
                qspan += len;
            }

            constexpr int32_t CIGAR_OP_CONSUMES_REF = 2;
            if (bam_cigar_type(op) & CIGAR_OP_CONSUMES_REF) {
                rspan += len;
            }
        }
    }

    Alignment ret{
            .qname = bam_get_qname(b),
            .qlen = b->core.l_qseq,
            .qstart = qstart,
            .qend = qstart + qspan,
            .strand = bam_is_rev(b) ? StrandOrientation::REVERSE : StrandOrientation::FORWARD,
            .rname = (b->core.tid >= 0) ? sam_hdr_tid2name(hdr, b->core.tid) : "*",
            .rlen = (b->core.tid >= 0) ? sam_hdr_tid2len(hdr, b->core.tid) : 0,
            .rstart = b->core.pos,
            .rend = b->core.pos + rspan,
            .mapq = b->core.qual,
            .flag = b->core.flag,
            .cigar = std::move(cigar_vec),
            .seq = utils::extract_sequence(b),
            .qual = utils::extract_quality(b),
            .dwell_stride = 0,
            .dwells = {},
    };

    std::tie(ret.dwell_stride, ret.dwells) = utils::extract_move_table(b);

    return ret;
}

}  // namespace dorado::secondary
