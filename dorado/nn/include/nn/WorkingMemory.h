#pragma once

#include <torch/types.h>

namespace dorado::nn {

enum class TensorLayout { NTC, TNC, CUTLASS_TNC_F16, CUTLASS_TNC_I8, CUBLAS_TN2C };
std::string to_string(const TensorLayout & layout);

// `WorkingMemory` encapsulates a backing tensor from which we create tensor views which map to
// either the front or the back of the backing tensor. The idea here is that we usually have one
// tensor with input data which we want to process to generate an output tensor. Once a processing
// step is done, the input tensor is no longer required and its memory can be reused, becoming the
// next output tensor. By creating views from alternating ends of one large tensor we can minimise
// the total amount of memory required.
//
// Sometimes the current tensor serves as both input and output of a processing step, but we also
// want a temporary buffer for the duration of the processing step. In that case `.temp()` can be
// called which creates a view of the specified size which will not be assigned to `.current`.
// A subsequent call to `.next_TC()` will create a view from the same end of the backing tensor,
// thus reusing the temp buffer memory.
//
// `.N`, the batch size, is constant for the lifetime of a `WorkingMemory` instance.
// `.T` (chunk size) and `.C` (channels) get updated with each call to `.next_TC()`
//
// Usage should be:
//   WorkingMemory wm(batch_size);
//   // Reservation phase, can mix `.next_TC()` and `.temp()`
//   wm.next_TC(chunk_size0, channels0, tensor_layout0);
//   wm.next_TC(chunk_size1, channels1, tensor_layout1);
//   wm.temp({dim0, ...}, dtype);
//   wm.next_TC(chunk_size2, channels2, tensor_layout2);
//    ...
//   wm.next_TC(chunk_sizeN, channelsN, tensor_layoutN);
//
//   // allocate_backing_tensor() begins use phase
//   wm.allocate_backing_tensor(device);
//
//   tensor0 = wm.next_TC(chunk_size0, channels0, tensor_layout0);
//    // write data to tensor0
//   tensor1 = wm.next_TC(chunk_size1, channels1, tensor_layout1);
//    // process: tensor0 -> tensor1
//   temp_tensor = wm.temp({dim0, ...}, dtype);
//    // process: tensor1 -> tensor1 with temp_tensor as temporary storage
//   tensor2 = wm.next_TC(chunk_size2, channels2, tensor_layout2);
//    // process: tensor1 -> tensor2
//    ...
//   tensorN = wm.next_TC(chunk_sizeN, channelsN, tensor_layoutN);
//    // process: tensorN-1 -> tensorN
//
// The pattern is: N calls to `.next_TC()/.temp()`, one call to `.allocate_backing_tensor()`,
// then N calls to `.next_TC()/.temp()` with the exact same parameters as before.
class WorkingMemory {
    // This may be overly conservative, but all CUDA allocation functions are guaranteed to
    // return 256-byte aligned pointers (even though GPU cache lines are at most 128 bytes).
    static constexpr int64_t ALIGNMENT = 256;

    static int64_t tensor_bytes(torch::IntArrayRef sizes, torch::Dtype dtype);
    at::Tensor next(torch::IntArrayRef sizes, torch::Dtype dtype, bool make_current);

public:
    explicit WorkingMemory(int batch_size) : N(batch_size) {}

    at::Tensor get_current_NTC_view();
    void next_N(int N_);
    at::Tensor next_TC(int T_, int C_, TensorLayout layout_);
    at::Tensor temp(torch::IntArrayRef sizes, torch::Dtype dtype);

    void allocate_backing_tensor(torch::Device dev);

    int64_t reservation_bytes{0};
    int64_t current_bytes{0};
    at::Tensor backing_tensor;
    at::Tensor current;  // The last tensor view created with `next(_, _, true)`
    TensorLayout layout{TensorLayout::NTC};
    bool is_input_to_rev_lstm{true};
    int N;     // batch size
    int T{0};  // current chunk size (time)
    int C{0};  // current layer size (channels)
};

}  // namespace dorado::nn
