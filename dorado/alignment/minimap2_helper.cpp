#include "minimap2_helper.h"

#include <minimap.h>

namespace dorado::alignment {

// Here mm_tbuf_t is used instead of mm_tbuf_s since minimap.h
// provides a typedef for mm_tbuf_s to mm_tbuf_t.
void MmTbufDestructor::operator()(mm_tbuf_t* tbuf) { mm_tbuf_destroy(tbuf); }

}  // namespace dorado::alignment
