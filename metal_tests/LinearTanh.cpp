#include "utils/metal_utils.h"

#include <catch2/catch.hpp>
#include <torch/torch.h>

#include <cstdint>

#define TEST_GROUP "Metal: "

TEST_CASE(TEST_GROUP "LinearTanh") {
    // Basic device setup.
    // get_mtl_device sets up an allocator that provides GPU/CPU shared memory
    // launch_kernel will create a MTL::CommandBuffer for us.
    MTL::Device *const device = get_mtl_device();
    REQUIRE(device != nullptr);
    MTL::CommandQueue *const command_queue = device->newCommandQueue();
    REQUIRE(command_queue != nullptr);

    // Note: Parameters below, e.g. layer_size, must correspond to a compiled shader.
    // If they don't, make_cps will fail.

    // Example values for HAC model run.
    const int layer_size = 384;  // LSTM layer size.  Set by the model architecture.
    const int batch_size = 192;  // Runtime-specified: number of chunks handled simultaneously.
    const int tile_size = 8;     // Size of simdgroup_* tiles.  Dictated by Metal itself.
    const int lstm_chunk_size = 1600;  // Number of samples in a chunk
    const int out_size = 1280;         // 4-mer -> 5-mer transition matrix => 4**4 * 5

    // Hardwired block size of 32x48 in the kernel imposes the constraint that the batch size be an
    // integral multiple of 48.
    assert(batch_size % 48 == 0);

    // This equates to the number of GPU cores.  16 is the figure for a complete M1 Pro.
    // We should probably test various values.
    const int kernel_thread_groups = 16;

    // This is determined from layer_size according to a hardwired table which
    // doesn't necessarily use the maximum possible.
    const int kernel_simd_groups = 24;

    // 32 threads/SIMD group on Apple GPUs.  The kernels have this hardwired.
    constexpr int kSimdGroupSize = 32;
    const int threads_per_thread_group = kernel_simd_groups * kSimdGroupSize;

    // Create a ComputePipelineState for the LinearTanh kernel.
    const std::string linear_fn = "linear_tanh_simd_" + std::to_string(layer_size) + "_fwd_" +
                                  std::to_string(kernel_simd_groups);
    MTL::ComputePipelineState *const linear_tanh_cps = make_cps(device, linear_fn);
    REQUIRE(linear_tanh_cps != nullptr);

    // Create a ComputePipelineState for the input reordering kernel.
    MTL::ComputePipelineState *const reorder_input_cps = make_cps(device, "reorder_input");
    REQUIRE(reorder_input_cps != nullptr);

    // Prepare args buffer.  Order in LstmArgs stuct:
    // layer_size (that is, the LSTM layer size)
    // reverse
    // chunk_tiles
    // chunk_size
    // linear_layer_size
    // This is following the MetalCRFModel way of doing things.
    const int kNumArgs = 5;
    const int32_t args_[] = {layer_size, 0, batch_size / tile_size, lstm_chunk_size, out_size};
    MTL::Buffer *const args = create_buffer(device, args_, kNumArgs);
    REQUIRE(args != nullptr);

    // Ensure we get the same random values for each run.
    torch::manual_seed(42);

    // We want the fake weights to be symmetrically distributed, or the output will be saturated.
    const torch::Tensor weights_f32 = torch::rand({layer_size, out_size}, torch::kFloat32) - 0.5f;
    const torch::Tensor biases_f32 = torch::rand({out_size}, torch::kFloat32) - 0.5f;

    // We combine the batch and chunk size dimensions into the leading dimension,
    // for input into the reorder_input kernel.
    const torch::Tensor in_f32 =
            torch::rand({lstm_chunk_size * batch_size, layer_size}, torch::kFloat32);

    // CPU comparison calculation (in float32 precision).
    const torch::Tensor out_cpu_f32 =
            5.0f * torch::tanh(torch::addmm(biases_f32, in_f32, weights_f32));

    // The kernel takes float16 weights, and works generally in float16.
    const torch::Tensor weights_f16 = weights_f32.to(torch::kFloat16);
    const torch::Tensor biases_f16 = biases_f32.to(torch::kFloat16);
    const torch::Tensor in_f16 = in_f32.to(torch::kFloat16);

    // Prepare weights/biases buffer for kernel.
    // The weights are a layer_size * out_size matrix, and come first in memory.
    // The biases are out_size entries, and immediately follow the weights
    // in memory.
    // If the weights matrix is viewed as layer_size rows each of size out_size,
    // then the biases form an additional row at the bottom.
    const torch::Tensor weights_biases_f16 =
            torch::cat({weights_f16, torch::unsqueeze(biases_f16, 0)}, 0);

    // Prepare the input buffer for the LinearTanh kernel.
    // reorder_inputs transforms the input in 3 ways:
    // 1) Rearranges input tiles in a fairly complex manner.
    // 2) Adds one time step of padding before and after the chunk time extents.
    // 3) Converts from float32 to float16.
    torch::Tensor in_f16_reordered =
            torch::zeros({(2 + lstm_chunk_size) * batch_size, layer_size}, torch::kFloat16);
    launch_kernel(reorder_input_cps, command_queue,
                  {args, mtl_for_tensor(in_f32), mtl_for_tensor(in_f16_reordered)},
                  kernel_thread_groups, threads_per_thread_group);

    // Perform the LinearTanh computation.
    torch::Tensor out_gpu_f32 =
            torch::zeros({lstm_chunk_size * batch_size, out_size}, torch::kFloat32);
    launch_kernel(linear_tanh_cps, command_queue,
                  {args, mtl_for_tensor(in_f16_reordered), mtl_for_tensor(weights_biases_f16),
                   mtl_for_tensor(out_gpu_f32)},
                  kernel_thread_groups, threads_per_thread_group);

    // Compare outputs.
    // The tolerances are somewhat arbitary, but we must account for GPU calculations in float16
    // versus CPU calculations in float32.
    // (The CPU calculation can be done in float16, but is too slow.)
    constexpr float kRelTolerance = 0.1f;
    constexpr float kAbsTolerance = 0.3f;
    REQUIRE(torch::allclose(out_cpu_f32, out_gpu_f32, kRelTolerance, kAbsTolerance));
    constexpr int kNormOrder = 1;  // Sum of absolute differences
    const float mean_sad =
            torch::linalg::vector_norm(out_cpu_f32 - out_gpu_f32, kNormOrder, {}, {}, {})
                    .item<float>() /
            out_cpu_f32.numel();
    REQUIRE(mean_sad < 0.005f);
}
