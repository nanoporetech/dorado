#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <map>
#include <string>
namespace dorado::utils {
std::map<std::string, std::shared_ptr<Read>> read_bam(std::string reads_file);
}