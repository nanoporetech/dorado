#pragma once

#include "torch_utils/metal_utils.h"

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::basecall {

struct MetalCallerTask;

using DecodedData = std::tuple<std::string, std::string, std::vector<uint8_t>>;

class MetalModel {
public:
    virtual ~MetalModel() = default;

    virtual at::Tensor create_input_tensor() const = 0;
    virtual bool call_task(MetalCallerTask& task,
                           std::mutex& inter_caller_mutex,
                           int try_count) = 0;
    virtual DecodedData decode(int chunk_idx) const = 0;
    virtual MTL::SharedEvent* decode_complete_event() const = 0;

protected:
    MetalModel() = default;
};

}  // namespace dorado::basecall
