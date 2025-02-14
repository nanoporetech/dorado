#include "types.h"

#include <ostream>

namespace dorado::correction {

std::ostream& operator<<(std::ostream& os, const OverlapWindow& ovl) {
    os << "overlap_idx = " << ovl.overlap_idx << ", win_tstart = " << ovl.win_tstart
       << ", win_tend = " << ovl.win_tend << ", tstart = " << ovl.tstart << ", tend = " << ovl.tend
       << ", qstart = " << ovl.qstart << ", qend = " << ovl.qend
       << ", cigar_start_idx = " << ovl.cigar_start_idx
       << ", cigar_start_offset = " << ovl.cigar_start_offset
       << ", cigar_end_idx = " << ovl.cigar_end_idx
       << ", cigar_end_offset = " << ovl.cigar_end_offset << ", accuracy = " << ovl.accuracy
       << ", columns = " << ovl.columns;
    return os;
}

}  // namespace dorado::correction
