#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <map>
#include <string>
namespace dorado::utils {
// Load a BAM/SAM/CRAM file and return a map of read_id -> dorado::Read
std::map<std::string, std::shared_ptr<Read>> read_bam(std::string reads_file);
}  // namespace dorado::utils
