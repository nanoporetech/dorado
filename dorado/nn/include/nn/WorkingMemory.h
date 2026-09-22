#pragma once

#include <ATen/core/TensorBody.h>

#include <string>

namespace dorado::nn {

enum class TensorLayout {
    NTC,
    NTC_I8,
    TNC,
    CUTLASS_TNC_F16,
    CUTLASS_TNC_I8,
    CUBLAS_TN2C,
    CUBLAS_TNC
};
std::string to_string(const TensorLayout & layout);

// `WorkingMemory` encapsulates a backing tensor from which we create tensor views which map to
// either the front or the back of the backing tensor, or the middle (for temporary buffers). The
// idea here is that we usually have one tensor with input data which we want to process, maybe
// using temporary buffers, to generate an output tensor. Once a processing step is done, the input
// tensor and any temporary tensors are no longer required and their memory can be reused, becoming
// the next output tensor. By creating views from alternating ends of one large tensor we can
// minimise the total amount of memory required.
//
// Any temporary buffers can be reserved with one or multiple calls to `.temp()`. After calling
// `.temp()` one or more times you MUST call either `.next_is_current()` (if input and output of
// the processing step are the same tensor) or `.next_NTC()` (if a new output tensor is required)
// afterwards.
//
//
// `.N`, the batch size, `.T` (chunk size) and `.C` (channels) can be specified with each call to
// `.next_NTC()`, though passing a negative value will keep the current value instead.
//
// Usage should be:
//   WorkingMemory wm;
//   // Reservation phase, use `.next_TC()`, `.temp()`, and `.next_is_current()`
//   wm.next_NTC(batch_size0, chunk_size0, channels0, tensor_layout0);
//   wm.temp({dim0, ...}, dtype);
//   wm.next_is_current();
//   wm.next_NTC(-1, chunk_size1, channels1, tensor_layout1);
//   wm.temp({dim1, ...}, dtype);
//   wm.next_NTC(-1, chunk_size2, channels2, tensor_layout2);
//    ...
//   wm.next_NTC(-1, chunk_sizeN, channelsN, tensor_layoutN);
//
//   // allocate_backing_tensor() begins use phase
//   wm.allocate_backing_tensor(device);
//
//   tensor0 = wm.next_NTC(batch_size0, chunk_size0, channels0, tensor_layout0);
//    // write data to tensor0
//   temp0 = wm.temp({dim0, ...}, dtype);
//   wm.next_is_current();
//    // process: tensor0 -> tensor0, with temp0 as temporary storage
//   tensor1 = wm.next_NTC(-1, chunk_size1, channels1, tensor_layout1);
//    // process: tensor0 -> tensor1
//   temp1 = wm.temp({dim1, ...}, dtype);
//   tensor2 = wm.next_NTC(-1, chunk_size2, channels2, tensor_layout2);
//    // process: tensor1 -> tensor2, with temp1 as temporary storage
//    ...
//   tensorN = wm.next_NTC(-1, chunk_sizeN, channelsN, tensor_layoutN);
//    // process: tensorN-1 -> tensorN
//
// The pattern is: N calls to `.next_NTC()/.temp()/.next_is_current()`, one call to
// `.allocate_backing_tensor()`, then N calls to `.next_NTC()/.temp().next_is_current()` with the
// exact same parameters as before.
class WorkingMemory {
    // This may be overly conservative, but all CUDA allocation functions are guaranteed to
    // return 256-byte aligned pointers (even though GPU cache lines are at most 128 bytes).
    static constexpr int64_t ALIGNMENT = 256;

    static int64_t tensor_bytes(at::IntArrayRef sizes, at::ScalarType dtype);
    at::Tensor alloc(at::IntArrayRef sizes, at::ScalarType dtype, bool make_current);

public:
    at::Tensor get_current_NTC_view();
    at::Tensor next_NTC(int N_, int T_, int C_, TensorLayout layout_);
    at::Tensor temp(at::IntArrayRef sizes, at::ScalarType dtype);
    at::Tensor next_is_current();

    void allocate_backing_tensor(c10::Device dev);

    int64_t reservation_bytes{0};
    int64_t current_bytes{0};
    int64_t temp_bytes{0};
    at::Tensor backing_tensor;
    at::Tensor current;  // The last tensor view created with `next_TC(...)`
    TensorLayout layout{TensorLayout::NTC};
    bool is_input_to_rev_lstm{true};
    int N{0};  // batch size
    int T{0};  // current chunk size (time)
    int C{0};  // current layer size (channels)
};

}  // namespace dorado::nn
