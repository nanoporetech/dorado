#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/types.h"


namespace dorado {

class Trimmer {
public:
    static BamPtr trim_barcode(BamPtr irecord, const BarcodeScoreResult& res, int seqlen);
    static void trim_barcode(SimplexRead& read, std::pair<int, int> interval);
    static std::pair<int, int> determine_trim_interval(const BarcodeScoreResult& res, int seqlen);
};

}  // namespace dorado
