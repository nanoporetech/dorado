#pragma once

#include "CallerModel.h"

#include <ATen/core/TensorBody.h>

namespace dorado::basecall {

enum class TensorLayout { NCT, NTC };

class TensorRunnerState final : public RunnerState {
public:
    TensorRunnerState(at::Tensor input, TensorLayout layout);

    void accept_chunk(int chunk_idx, const at::Tensor& chunk_CT) override;
    size_t chunk_size() const override;
    size_t batch_size() const override;

    at::Tensor& input() { return m_input; }
    const at::Tensor& input() const { return m_input; }

private:
    at::Tensor m_input;
    TensorLayout m_layout;
};

}  // namespace dorado::basecall
