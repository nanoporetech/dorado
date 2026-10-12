#include "caller/TensorRunnerState.h"

#include <ATen/TensorIndexing.h>  // IWYU pragma: keep for .index
#include <ATen/ops/arange.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[TensorRunnerState]"

namespace {

at::Tensor make_chunk() {
    return at::arange(12, at::TensorOptions().dtype(at::kFloat)).view({3, 4});
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " stores CT chunks in NCT tensors", CUT_TAG) {
    auto input = at::zeros({2, 3, 4});
    dorado::basecall::TensorRunnerState state(input, dorado::basecall::TensorLayout::NCT);
    const auto chunk = make_chunk();

    state.accept_chunk(1, chunk);

    CATCH_CHECK(state.batch_size() == 2);
    CATCH_CHECK(state.chunk_size() == 4);
    CATCH_CHECK(at::equal(state.input().index({1}), chunk));
    CATCH_CHECK(at::equal(state.input().index({0}), at::zeros_like(chunk)));
}

CATCH_TEST_CASE(CUT_TAG " transposes CT chunks into NTC tensors", CUT_TAG) {
    auto input = at::zeros({2, 4, 3});
    dorado::basecall::TensorRunnerState state(input, dorado::basecall::TensorLayout::NTC);
    const auto chunk = make_chunk();

    state.accept_chunk(1, chunk);

    CATCH_CHECK(state.batch_size() == 2);
    CATCH_CHECK(state.chunk_size() == 4);
    CATCH_CHECK(at::equal(state.input().index({1}), chunk.transpose(0, 1)));
    CATCH_CHECK(at::equal(state.input().index({0}), at::zeros_like(chunk.transpose(0, 1))));
}
