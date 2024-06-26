#pragma once

struct mm_tbuf_s;

namespace dorado::alignment {

struct MmTbufDestructor {
    void operator()(mm_tbuf_s *);
};
using MmTbufPtr = std::unique_ptr<mm_tbuf_s, MmTbufDestructor>;

}  // namespace dorado::alignment