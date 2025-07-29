#include "hts_utils/FastxSequentialReader.h"

#include <htslib/faidx.h>
#include <spdlog/spdlog.h>
#include <zlib.h>

#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146)  // unary minus on unsigned
#pragma warning(disable : 4244)  // conversion from 'int' to 'char'
#pragma warning(disable : 4267)  // conversion from 'size_t' to 'int'
#endif

#include <htslib/kseq.h>

KSEQ_INIT(gzFile, gzread)

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace dorado::hts_io {

struct FastxSequentialReader::Data {
    gzFile fp{nullptr};
    kseq_t* seq{nullptr};
};

FastxSequentialReader::FastxSequentialReader(const std::filesystem::path& in_path)
        : fmt_{hts_io::parse_sequence_format(in_path)},
          data_{std::make_unique<FastxSequentialReader::Data>()} {
    data_->fp = gzopen(in_path.string().c_str(), "r");
    if (!data_->fp) {
        throw std::runtime_error{"Could not open file: " + in_path.string()};
    }
    data_->seq = kseq_init(data_->fp);
}

FastxSequentialReader::~FastxSequentialReader() {
    if (data_->seq) {
        kseq_destroy(data_->seq);
    }
    if (data_->fp) {
        gzclose(data_->fp);
    }
}

bool FastxSequentialReader::get_next(FastxRecord& record) {
    while (kseq_read(data_->seq) >= 0) {
        record = FastxRecord{
                .name = std::string_view(data_->seq->name.s, data_->seq->name.l),
                .comment = std::string_view(data_->seq->comment.s, data_->seq->comment.l),
                .seq = std::string_view(data_->seq->seq.s, data_->seq->seq.l),
                .qual = std::string_view(data_->seq->qual.s, data_->seq->qual.l),
        };
        return true;
    }
    return false;
}

hts_io::SequenceFormatType FastxSequentialReader::get_format() const { return fmt_; }

}  // namespace dorado::hts_io
