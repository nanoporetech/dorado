#include "types.h"

#include "htslib/sam.h"

namespace dorado {

void BamDestructor::operator()(bam1_t *bam) { bam_destroy1(bam); }

}  // namespace dorado
