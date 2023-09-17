#include "read_utils.h"

namespace dorado::utils {
ReadPtr shallow_copy_read(const Read& read) {
    auto copy = ReadPtr::make();
    copy->read_common.raw_data = read.read_common.raw_data;
    copy->digitisation = read.digitisation;
    copy->range = read.range;
    copy->offset = read.offset;
    copy->sample_rate = read.sample_rate;

    copy->shift = read.shift;
    copy->scale = read.scale;

    copy->scaling = read.scaling;

    copy->read_common.model_stride = read.read_common.model_stride;

    copy->read_common.read_id = read.read_common.read_id;
    copy->read_common.seq = read.read_common.seq;
    copy->read_common.qstring = read.read_common.qstring;
    copy->read_common.moves = read.read_common.moves;
    copy->read_common.run_id = read.read_common.run_id;
    copy->read_common.model_name = read.read_common.model_name;

    copy->read_common.base_mod_probs = read.read_common.base_mod_probs;
    copy->mod_base_info = read.mod_base_info;

    copy->num_trimmed_samples = read.num_trimmed_samples;

    copy->attributes = read.attributes;

    copy->start_sample = read.start_sample;
    copy->end_sample = read.end_sample;
    copy->run_acquisition_start_time_ms = read.run_acquisition_start_time_ms;
    copy->is_duplex = read.is_duplex;
    return copy;
}

}  // namespace dorado::utils
