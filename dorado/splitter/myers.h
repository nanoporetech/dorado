#pragma once

#include <iosfwd>
#include <string_view>
#include <vector>

namespace dorado::splitter {

struct EdistResult {
    std::size_t begin;  // inclusive
    std::size_t end;    // exclusive
    std::size_t edist;
};
std::vector<EdistResult> myers_align(std::string_view query,
                                     std::string_view seq,
                                     std::size_t max_edist);

void print_edists(std::ostream& os, std::string_view seq, const std::vector<size_t>& edists);

}  // namespace dorado::splitter
