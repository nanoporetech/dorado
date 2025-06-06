#pragma once
#include "utils/types.h"

#include <utility>

struct bam1_t;

namespace dorado {

class SimplexRead;
struct AdapterScoreResult;
struct BarcodeScoreResult;

class Trimmer {
public:
    static BamPtr trim_sequence(bam1_t* irecord, std::pair<int, int> interval);
    static void trim_sequence(SimplexRead& read, std::pair<int, int> interval);
    static std::pair<int, int> determine_trim_interval(const BarcodeScoreResult& res, int seqlen);
    static std::pair<int, int> determine_trim_interval(AdapterScoreResult& res, int seqlen);
    static void check_and_update_barcoding(SimplexRead& read);
};

}  // namespace dorado
