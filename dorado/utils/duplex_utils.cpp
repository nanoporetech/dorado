#include "duplex_utils.h"

#include <algorithm>
#include <fstream>
#include <vector>

namespace dorado::utils {
std::map<std::string, std::string> load_pairs_file(std::string pairs_file) {
    std::ifstream dataFile;
    dataFile.open(pairs_file);

    std::string cell;
    int line = 0;

    std::getline(dataFile, cell);
    std::map<std::string, std::string> template_complement_map;
    while (!dataFile.eof()) {
        char delim = ' ';
        auto delim_pos = cell.find(delim);

        std::string t = cell.substr(0, delim_pos);
        std::string c = cell.substr(delim_pos + 1, delim_pos * 2 - 1);
        template_complement_map[t] = c;

        line++;
        std::getline(dataFile, cell);
    }
    return template_complement_map;
}

void reverse_complement(std::vector<char>& sequence) {
    std::reverse(sequence.begin(), sequence.end());
    std::map<char, char> complementary_nucleotides = {
            {'A', 'T'}, {'C', 'G'}, {'G', 'C'}, {'T', 'A'}};
    std::for_each(sequence.begin(), sequence.end(),
                  [&complementary_nucleotides](char& c) { c = complementary_nucleotides[c]; });
}
}  // namespace dorado::utils