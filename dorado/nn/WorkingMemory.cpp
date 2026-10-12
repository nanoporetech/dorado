#include "nn/WorkingMemory.h"

#include "utils/math_utils.h"

#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>

#include <stdexcept>

namespace dorado::nn {

std::string to_string(const TensorLayout &layout) {
    switch (layout) {
    case TensorLayout::NTC:
        return std::string("NTC");
    case TensorLayout::NTC_I8:
        return std::string("NTC_I8");
    case TensorLayout::TNC:
        return std::string("TNC");
    case TensorLayout::CUTLASS_TNC_F16:
        return std::string("CUTLASS_TNC_F16");
    case TensorLayout::CUTLASS_TNC_I8:
        return std::string("CUTLASS_TNC_I8");
    case TensorLayout::CUBLAS_TN2C:
        return std::string("CUBLAS_TN2C");
    case TensorLayout::CUBLAS_TNC:
        return std::string("CUBLAS_TNC");
    }
    throw std::logic_error("TensorLayout unknown");
}

int64_t WorkingMemory::tensor_bytes(at::IntArrayRef sizes, at::ScalarType dtype) {
    auto elems = c10::multiply_integers(sizes);
    return utils::pad_to<int64_t>(elems * at::elementSize(dtype), ALIGNMENT);
}

at::Tensor WorkingMemory::alloc(at::IntArrayRef sizes, at::ScalarType dtype, bool make_current) {
    const int64_t new_bytes = tensor_bytes(sizes, dtype);
    at::Tensor new_tensor;
    const int64_t bytes_in_use = current_bytes + temp_bytes + new_bytes;
    if (!backing_tensor.defined()) {
        // If no backing tensor is allocated yet we're still in the reservation phase
        reservation_bytes = std::max(reservation_bytes, bytes_in_use);
    } else {
        if (bytes_in_use > reservation_bytes) {
            throw std::runtime_error("WorkingMemory: overlap detected.");
        }

        const bool current_is_front =
                current.defined() && current.data_ptr() == backing_tensor.data_ptr();
        auto elems = c10::multiply_integers(sizes);
        const auto bt_dtype = backing_tensor.view(dtype);

        // if `!make_current`, add a new temp tensor after `current` and any existing temp tensors
        const int64_t skip_bytes = make_current ? 0ll : current_bytes + temp_bytes;

        // count from front or from back?
        const int64_t start_byte = (current_is_front == make_current)
                                           ? reservation_bytes - skip_bytes - new_bytes
                                           : skip_bytes;

        new_tensor = bt_dtype.narrow(0, start_byte / at::elementSize(dtype), elems).view(sizes);
    }
    if (make_current) {
        current_bytes = new_bytes;
        current = new_tensor;
        temp_bytes = 0;
    } else {
        temp_bytes += new_bytes;
    }
    return new_tensor;
}

at::Tensor WorkingMemory::get_current_NTC_view() {
    switch (layout) {
    case TensorLayout::NTC:
    case TensorLayout::NTC_I8:
        return current;
    case TensorLayout::TNC:
        return current.transpose(0, 1);
    case TensorLayout::CUTLASS_TNC_F16:
    case TensorLayout::CUTLASS_TNC_I8:
        return current.narrow(0, is_input_to_rev_lstm ? 1 : 2, T).transpose(1, 0);
    case TensorLayout::CUBLAS_TN2C:
        return current.narrow(0, is_input_to_rev_lstm ? 1 : 0, T)
                .transpose(1, 0)
                .select(2, is_input_to_rev_lstm ? 1 : 0);
    case TensorLayout::CUBLAS_TNC:
        return current.narrow(0, is_input_to_rev_lstm ? 1 : 2, T).transpose(1, 0);
    default:
        throw std::logic_error("Unhandled TensorLayout");
    }
}

at::Tensor WorkingMemory::next_NTC(int N_, int T_, int C_, TensorLayout layout_) {
    N = (N_ > 0) ? N_ : N;
    T = (T_ > 0) ? T_ : T;
    C = (C_ > 0) ? C_ : C;
    layout = layout_;
    if (layout == TensorLayout::NTC) {
        return alloc({N, T, C}, at::kHalf, true);
    } else if (layout == TensorLayout::NTC_I8) {
        return alloc({N, T, C}, at::kChar, true);
    } else if (layout == TensorLayout::TNC) {
        return alloc({T, N, C}, at::kHalf, true);
    } else if (layout == TensorLayout::CUTLASS_TNC_F16) {
        return alloc({T + 5, N, C}, at::kHalf, true);
    } else if (layout == TensorLayout::CUTLASS_TNC_I8) {
        return alloc({T + 5, N, C}, at::kChar, true);
    } else if (layout == TensorLayout::CUBLAS_TN2C) {
        return alloc({T + 1, N, 2, C}, at::kHalf, true);
    } else if (layout == TensorLayout::CUBLAS_TNC) {
        return alloc({T + 3, N, C}, at::kHalf, true);
    } else {
        throw std::logic_error("Unhandled TensorLayout");
    }
}

at::Tensor WorkingMemory::next_is_current() {
    // release temp buffers of the current pass
    temp_bytes = 0;
    return current;
}

at::Tensor WorkingMemory::temp(at::IntArrayRef sizes, at::ScalarType dtype) {
    return alloc(sizes, dtype, false);
}

void WorkingMemory::allocate_backing_tensor(c10::Device dev) {
    if (!backing_tensor.defined()) {
        // Using kF16 here because the libtorch version on TX2 doesn't support `Tensor::view()`
        // with a dtype of a different size, and all buffers are kF16 on TX2.
        backing_tensor = at::empty({reservation_bytes / 2},
                                   at::TensorOptions().device(dev).dtype(at::kHalf));
    }
    current_bytes = 0;
    temp_bytes = 0;
}

}  // namespace dorado::nn
