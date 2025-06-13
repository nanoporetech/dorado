#pragma once

#include "splitter/ReadSplitter.h"

#include <ATen/core/TensorBody.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace dorado::splitter {

template <class FilterF>
auto filter_ranges(const PosRanges& ranges, FilterF filter_f) {
    PosRanges filtered;
    std::copy_if(ranges.begin(), ranges.end(), std::back_inserter(filtered), filter_f);
    return filtered;
}

//merges overlapping ranges and ranges separated by merge_dist or less
//ranges supposed to be sorted by start coordinate
PosRanges merge_ranges(const PosRanges& ranges, uint64_t merge_dist);

SimplexReadPtr subread(const SimplexRead& read,
                       std::optional<PosRange> seq_range,
                       PosRange signal_range);

template <typename T>
SampleRanges<T> detect_pore_signal(const at::Tensor& signal,
                                   T threshold,
                                   uint64_t cluster_dist,
                                   uint64_t ignore_prefix) {
    SampleRanges<T> ans;
    auto pore_a = signal.accessor<T, 1>();
    int64_t cl_start = -1;
    int64_t cl_end = -1;

    T cl_max = std::numeric_limits<T>::min();
    int64_t cl_argmax = -1;
    for (auto i = ignore_prefix; i < uint64_t(pore_a.size(0)); i++) {
        if (pore_a[i] > threshold) {
            //check if we need to start new cluster
            if (cl_end == -1 || i > cl_end + cluster_dist) {
                //report previous cluster
                if (cl_end != -1) {
                    assert(cl_start != -1);
                    ans.push_back(SampleRange(cl_start, cl_end, cl_argmax, cl_max));
                }
                cl_start = i;
                cl_max = std::numeric_limits<T>::min();
            }
            if (pore_a[i] >= cl_max) {
                cl_max = pore_a[i];
                cl_argmax = i;
            }
            cl_end = i + 1;
        }
    }
    //report last cluster
    if (cl_end != -1) {
        assert(cl_start != -1);
        assert(cl_start < pore_a.size(0) && cl_end <= pore_a.size(0));
        ans.push_back(SampleRange(cl_start, cl_end, cl_argmax, cl_max));
    }

    return ans;
}

}  // namespace dorado::splitter
