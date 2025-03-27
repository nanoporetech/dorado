#include "types.h"

#include "torch_utils/tensor_utils.h"

#include <ostream>
#include <tuple>

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

std::ostream& operator<<(std::ostream& os, const WindowFeatures& wf) {
    os << "bases = " << utils::tensor_shape_as_string(wf.bases)
       << ", quals = " << utils::tensor_shape_as_string(wf.quals)
       << ", indices = " << utils::tensor_shape_as_string(wf.indices) << ", length = " << wf.length
       << ", supported.size = " << std::size(wf.supported)
       << ", inferred_bases.size = " << std::size(wf.inferred_bases) << ", n_alns = " << wf.n_alns
       << ", read_name = " << wf.read_name << ", window_idx = " << wf.window_idx;

    return os;
}

bool operator==(const OverlapWindow& lhs, const OverlapWindow& rhs) {
    return std::tie(lhs.overlap_idx, lhs.win_tstart, lhs.win_tend, lhs.tstart, lhs.tend, lhs.qstart,
                    lhs.qend, lhs.cigar_start_idx, lhs.cigar_start_offset, lhs.cigar_end_idx,
                    lhs.cigar_end_offset, lhs.accuracy, lhs.columns) ==
           std::tie(rhs.overlap_idx, rhs.win_tstart, rhs.win_tend, rhs.tstart, rhs.tend, rhs.qstart,
                    rhs.qend, rhs.cigar_start_idx, rhs.cigar_start_offset, rhs.cigar_end_idx,
                    rhs.cigar_end_offset, rhs.accuracy, rhs.columns);
}

}  // namespace dorado::correction
