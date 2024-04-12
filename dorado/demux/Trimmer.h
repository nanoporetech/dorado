#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/types.h"

namespace dorado {

class Trimmer {
public:
    static BamPtr trim_sequence(bam1_t* irecord, std::pair<int, int> interval);
    static void trim_sequence(SimplexRead& read, std::pair<int, int> interval);
    static std::pair<int, int> determine_trim_interval(const BarcodeScoreResult& res, int seqlen);
    static std::pair<int, int> determine_trim_interval(const AdapterScoreResult& res, int seqlen);
};

}  // namespace dorado
