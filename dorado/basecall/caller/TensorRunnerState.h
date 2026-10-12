#pragma once

#include <ATen/core/TensorBody.h>

namespace dorado::basecall {

enum class TensorLayout { NCT, NTC };

class TensorRunnerState final {
public:
    TensorRunnerState(at::Tensor input, TensorLayout layout);

    void accept_chunk(int chunk_idx, const at::Tensor& chunk_CT);
    size_t chunk_size() const;
    size_t batch_size() const;
    bool variable_chunk_sizes() const { return false; }

    at::Tensor& input() { return m_input; }
    const at::Tensor& input() const { return m_input; }

private:
    at::Tensor m_input;
    TensorLayout m_layout;
};

}  // namespace dorado::basecall
