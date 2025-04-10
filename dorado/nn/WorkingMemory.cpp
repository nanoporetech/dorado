#include "WorkingMemory.h"

#include "utils/math_utils.h"

#if DORADO_CUDA_BUILD
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace dorado::nn {

#if DORADO_CUDA_BUILD

std::string to_string(const TensorLayout &layout) {
    switch (layout) {
    case TensorLayout::NTC:
        return std::string("NTC");
    case TensorLayout::TNC:
        return std::string("TNC");
    case TensorLayout::CUTLASS_TNC_F16:
        return std::string("CUTLASS_TNC_F16");
    case TensorLayout::CUTLASS_TNC_I8:
        return std::string("CUTLASS_TNC_I8");
    case TensorLayout::CUBLAS_TN2C:
        return std::string("CUBLAS_TN2C");
    default:
        return std::string("TensorLayout::UNKNOWN");
    }
}

int64_t WorkingMemory::tensor_bytes(torch::IntArrayRef sizes, torch::Dtype dtype) {
    auto elems = c10::multiply_integers(sizes);
    return utils::pad_to<int64_t>(elems * torch::elementSize(dtype), ALIGNMENT);
}

at::Tensor WorkingMemory::next(torch::IntArrayRef sizes, torch::Dtype dtype, bool make_current) {
    auto new_bytes = tensor_bytes(sizes, dtype);
    at::Tensor new_tensor;
    if (!backing_tensor.defined()) {
        // If no backing tensor is allocated yet we're still in the reservation phase
        reservation_bytes = std::max(reservation_bytes, current_bytes + new_bytes);
    } else {
        if (current_bytes + new_bytes > reservation_bytes) {
            throw std::runtime_error("WorkingMemory: overlap detected.");
        }

        bool current_is_front =
                current.defined() && current.data_ptr() == backing_tensor.data_ptr();
        auto elems = c10::multiply_integers(sizes);
        auto bt_dtype = backing_tensor.view(dtype);
        auto start_pos = current_is_front
                                 ? (reservation_bytes - new_bytes) / torch::elementSize(dtype)
                                 : int64_t(0);
        new_tensor = bt_dtype.narrow(0, start_pos, elems).view(sizes);
    }
    if (make_current) {
        current_bytes = new_bytes;
        current = new_tensor;
    }
    return new_tensor;
}

at::Tensor WorkingMemory::get_current_NTC_view() {
    switch (layout) {
    case TensorLayout::NTC:
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
    default:
        throw std::logic_error("Unhandled TensorLayout");
    }
}

at::Tensor WorkingMemory::next_TC(int T_, int C_, TensorLayout layout_) {
    T = T_;
    C = C_;
    layout = layout_;
    if (layout == TensorLayout::NTC) {
        return next({N, T, C}, torch::kF16, true);
    } else if (layout == TensorLayout::TNC) {
        return next({T, N, C}, torch::kF16, true);
    } else if (layout == TensorLayout::CUTLASS_TNC_F16) {
        return next({T + 3, N, C}, torch::kF16, true);
    } else if (layout == TensorLayout::CUTLASS_TNC_I8) {
        return next({T + 3, N, C}, torch::kI8, true);
    } else if (layout == TensorLayout::CUBLAS_TN2C) {
        return next({T + 1, N, 2, C}, torch::kF16, true);
    } else {
        throw std::logic_error("Unhandled TensorLayout");
    }
}

at::Tensor WorkingMemory::temp(torch::IntArrayRef sizes, torch::Dtype dtype) {
    return next(sizes, dtype, false);
}

void WorkingMemory::allocate_backing_tensor(torch::Device dev) {
    // Using kF16 here because the libtorch version on TX2 doesn't support `Tensor::view()`
    // with a dtype of a different size, and all buffers are kF16 on TX2.
    backing_tensor = torch::empty({reservation_bytes / 2},
                                  at::TensorOptions().device(dev).dtype(torch::kF16));
    current_bytes = 0;
}

#endif  // if DORADO_CUDA_BUILD

}  // namespace dorado::nn