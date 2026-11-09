#include "secondary/common/resolve_model.h"

#include "utils/log_utils.h"
#include "utils/string_utils.h"

#include <algorithm>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace dorado::secondary {

void print_basecaller_models(std::ostream& os,
                             const std::unordered_set<std::string>& basecaller_models,
                             const std::string& delimiter) {
    std::vector<std::string> lines(std::begin(basecaller_models), std::end(basecaller_models));
    std::sort(std::begin(lines), std::end(lines));
    os << utils::join(lines, delimiter);
}

void throw_unless_single_model_present(const BamInfo& bam_info) {
    // Check that there is not exactly one basecaller listed in the BAM. Otherwise, no auto resolving.
    if (std::size(bam_info.basecaller_models) != 1) {
        if (std::empty(bam_info.basecaller_models)) {
            throw std::runtime_error{
                    "Input BAM file has no basecaller models listed in the header."};
        }
        if (std::size(bam_info.basecaller_models) > 1) {
            std::ostringstream oss;
            oss << "Input BAM file has a mix of different basecaller models. Only one basecaller "
                   "model can be processed. List of all basecaller models found in the BAM file: ";
            print_basecaller_models(oss, bam_info.basecaller_models, ", ");
            throw std::runtime_error{oss.str()};
        }
    }
}

void check_stereo_models(const BamInfo& bam_info, const bool throw_error) {
    // Check if any of the input models is a stereo, to report a clear error that this is not supported.
    for (const std::string& model : bam_info.basecaller_models) {
        if (model.find("stereo") != std::string::npos) {
            std::ostringstream oss;
            oss << "Inputs from duplex basecalling are not supported. Detected model: '" << model
                << "' in the input BAM.";
            if (throw_error) {
                throw std::runtime_error{oss.str()};
            } else {
                DORADO_LOG(warn, "{} This may produce inferior results.", oss.str());
            }
        }
    }
}

}  // namespace dorado::secondary
