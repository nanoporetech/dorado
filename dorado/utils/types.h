#pragma once

#include <array>
#include <memory>
#include <string>

struct bam1_t;
struct mm_tbuf_s;

namespace dorado {

struct ReadGroup {
    std::string run_id;
    std::string basecalling_model;
    std::string flowcell_id;
    std::string device_id;
    std::string exp_start_time;
    std::string sample_id;
};

struct BamDestructor {
    void operator()(bam1_t *);
};
using BamPtr = std::unique_ptr<bam1_t, BamDestructor>;

struct MmTbufDestructor {
    void operator()(mm_tbuf_s *);
};
using MmTbufPtr = std::unique_ptr<mm_tbuf_s, MmTbufDestructor>;

enum class ReadOrder { UNRESTRICTED, BY_CHANNEL, BY_TIME };

inline std::string to_string(ReadOrder read_order) {
    switch (read_order) {
    case ReadOrder::UNRESTRICTED:
        return "UNRESTRICTED";
    case ReadOrder::BY_CHANNEL:
        return "BY_CHANNEL";
    case ReadOrder::BY_TIME:
        return "BY_TIME";
    default:
        return "Unknown";
    }
}

struct ModBaseInfo {
    ModBaseInfo() = default;
    ModBaseInfo(std::string alphabet_, std::string long_names_, std::string context_)
            : alphabet(std::move(alphabet_)),
              long_names(std::move(long_names_)),
              context(std::move(context_)) {}
    std::string alphabet;
    std::string long_names;
    std::string context;
    std::array<size_t, 4> base_counts{};
};

}  // namespace dorado
