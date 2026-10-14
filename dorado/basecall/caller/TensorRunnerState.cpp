#include "TensorRunnerState.h"

#include <ATen/TensorIndexing.h>  // IWYU pragma: keep for at::indexing::TensorIndex in index_put_

#include <cassert>
#include <utility>

namespace dorado::basecall {

TensorRunnerState::TensorRunnerState(at::Tensor input, TensorLayout layout)
        : m_input(std::move(input)), m_layout(layout) {
    assert(m_input.dim() == 3);
}

void TensorRunnerState::accept_chunk(int chunk_idx, const at::Tensor& chunk_CT) {
    assert(chunk_CT.dim() == 2);
    auto chunk_matched_layout =
            (m_layout == TensorLayout::NTC) ? chunk_CT.transpose(0, 1) : chunk_CT;
    assert(m_input.size(1) == chunk_matched_layout.size(0));
    assert(m_input.size(2) == chunk_matched_layout.size(1));
    m_input.index_put_({chunk_idx}, chunk_matched_layout);
}

size_t TensorRunnerState::chunk_size() const {
    return m_layout == TensorLayout::NTC ? m_input.size(1) : m_input.size(2);
}

size_t TensorRunnerState::batch_size() const { return m_input.size(0); }

}  // namespace dorado::basecall
