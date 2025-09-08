#pragma once

#include <ATen/ATen.h>

#include <string>
#include <tuple>
#include <vector>

namespace dorado::secondary {

std::tuple<at::Tensor, std::vector<std::string>> reorder_chunk(
        const at::Tensor& chunk,
        const std::vector<std::string>& prev_rids_out,
        const std::vector<std::string>& rids_in,
        const std::vector<std::string>& rids_out);

}  // namespace dorado::secondary
