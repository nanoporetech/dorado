#include "read_utils.h"

namespace dorado::utils {
SimplexReadPtr shallow_copy_read(const SimplexRead& read) {
    auto copy = std::make_unique<SimplexRead>();
    copy->read_common.raw_data = read.read_common.raw_data;
    copy->digitisation = read.digitisation;
    copy->range = read.range;
    copy->offset = read.offset;
    copy->read_common.sample_rate = read.read_common.sample_rate;

    copy->read_common.shift = read.read_common.shift;
    copy->read_common.scale = read.read_common.scale;

    copy->scaling = read.scaling;

    copy->read_common.model_stride = read.read_common.model_stride;

    copy->read_common.read_id = read.read_common.read_id;
    copy->read_common.seq = read.read_common.seq;
    copy->read_common.qstring = read.read_common.qstring;
    copy->read_common.moves = read.read_common.moves;
    copy->read_common.run_id = read.read_common.run_id;
    copy->read_common.flowcell_id = read.read_common.flowcell_id;
    copy->read_common.position_id = read.read_common.position_id;
    copy->read_common.experiment_id = read.read_common.experiment_id;
    copy->read_common.model_name = read.read_common.model_name;

    copy->read_common.base_mod_probs = read.read_common.base_mod_probs;
    copy->read_common.mod_base_info = read.read_common.mod_base_info;

    copy->read_common.num_trimmed_samples = read.read_common.num_trimmed_samples;

    copy->read_common.attributes = read.read_common.attributes;

    copy->start_sample = read.start_sample;
    copy->end_sample = read.end_sample;
    copy->run_acquisition_start_time_ms = read.run_acquisition_start_time_ms;
    copy->read_common.is_duplex = read.read_common.is_duplex;
    return copy;
}

}  // namespace dorado::utils
