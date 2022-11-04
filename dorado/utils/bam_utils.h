#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <map>
#include <string>

std::map<std::string, std::shared_ptr<Read>> read_bam(std::string reads_file);