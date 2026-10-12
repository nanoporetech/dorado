#include "TensorRunnerState.h"

#include <ATen/TensorIndexing.h>

#include <cassert>
#include <utility>

namespace dorado::basecall {

TensorRunnerState::TensorRunnerState(at::Tensor input, TensorLayout layout)
        : m_input(std::move(input)), m_layout(layout) {
    assert(m_input.dim() == 3);
}

void TensorRunnerState::accept_chunk(int chunk_idx, const at::Tensor& chunk_CT) {
    assert(chunk_CT.dim() == 2);
    using at::indexing::Ellipsis;
    if (m_layout == TensorLayout::NTC) {
        assert(m_input.size(1) == chunk_CT.size(1));
        assert(m_input.size(2) == chunk_CT.size(0));
        m_input.index_put_({chunk_idx, Ellipsis, Ellipsis}, chunk_CT.transpose(0, 1));
    } else {
        assert(m_input.size(1) == chunk_CT.size(0));
        assert(m_input.size(2) == chunk_CT.size(1));
        m_input.index_put_({chunk_idx, Ellipsis, Ellipsis}, chunk_CT);
    }
}

size_t TensorRunnerState::chunk_size() const {
    return m_layout == TensorLayout::NTC ? m_input.size(1) : m_input.size(2);
}

size_t TensorRunnerState::batch_size() const { return m_input.size(0); }

}  // namespace dorado::basecall
