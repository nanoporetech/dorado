#include "torch_utils/metal_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace {

using namespace dorado::utils;

// For convenient Slice syntax.
using namespace torch::indexing;

namespace F = torch::nn::functional;

#define TEST_GROUP "Metal: "

float MeanAbsDiff(const at::Tensor &a, const at::Tensor &b) {
    CATCH_REQUIRE(a.numel() == b.numel());
    return torch::sum(torch::abs(a - b)).item<float>() / a.numel();
}

// 32 threads/SIMD group on Apple GPUs.  The kernels have this hardwired.
constexpr int kSimdGroupSize = 32;

}  // namespace

CATCH_TEST_CASE(TEST_GROUP "Linear") {
    // Basic device setup.
    // get_mtl_device sets up an allocator that provides GPU/CPU shared memory
    // launch_kernel will create a MTL::CommandBuffer for us.
    const NS::SharedPtr<MTL::Device> device = get_mtl_device();
    CATCH_REQUIRE(device);
    const NS::SharedPtr<MTL::CommandQueue> command_queue =
            NS::TransferPtr(device->newCommandQueue());
    CATCH_REQUIRE(command_queue);

    // Example values for HAC model run.
    const int layer_size = 384;       // Typical LSTM layer size for HAC model.
    const int in_batch_size = 96;     // Runtime-specified: number of chunks handled simultaneously.
    const int tile_size = 8;          // Size of simdgroup_* tiles.  Dictated by Metal itself.
    const int lstm_chunk_size = 200;  // Number of samples in a chunk divided by model stride
    const int out_size = 1024;  // 4-mer transition matrix without fixed stay scores => 4**4 * 4
    const int batch_split = 2;

    // Hardwired block size of 32x48 in the kernel imposes the constraint that the batch size be an
    // integral multiple of 48.
    assert(in_batch_size % 48 == 0);

    // This equates to the number of GPU cores.  16 is the figure for a complete M1 Pro.
    // We should probably test various values.
    const int kernel_thread_groups = 16;

    // This is determined from layer_size according to a hardwired table which
    // doesn't necessarily use the maximum possible.
    const int kernel_simd_groups = 24;

    const int threads_per_thread_group = kernel_simd_groups * kSimdGroupSize;

    // Threadgroup memory size calculation.
    constexpr int kTileSize = 8;
    typedef uint16_t ftype;
    const int kOutBufSize = sizeof(ftype) * kernel_simd_groups * kTileSize * kTileSize;
    const int kOutBufF32Size = sizeof(float) * kernel_simd_groups * kTileSize * kTileSize;
    const std::vector<int> tg_buffer_lens{kOutBufSize, kOutBufF32Size};

    // Create a ComputePipelineState for the input reordering kernel.
    const NS::SharedPtr<MTL::ComputePipelineState> reorder_input_cps =
            make_cps(device.get(), "reorder_input_to_rev_lstm_output",
                     {{"kLstmLayerSize", layer_size}}, std::nullopt);
    CATCH_REQUIRE(reorder_input_cps);

    // Order in LstmArgs struct (which is also used by reorder_input):
    // batch_tiles
    // chunk_size
    // time_step_begin
    // time_step_end
    const std::vector<int32_t> args_reorder_{in_batch_size / tile_size, lstm_chunk_size, 0, 0};
    const NS::SharedPtr<MTL::Buffer> args_reorder = create_vec_buffer(device.get(), args_reorder_);
    CATCH_REQUIRE(args_reorder);

    // Ensure we get the same random values for each run.
    torch::manual_seed(42);

    // The kernel takes weights and biases in a single tensor,
    // We want the fake weights to be symmetrically distributed, or the output will be saturated.
    const at::Tensor weights_biases_f32 =
            torch::rand({layer_size + 1, out_size}, torch::kFloat32) - 0.5f;
    const at::Tensor weights_f32 = weights_biases_f32.slice(0, 0, layer_size);
    const at::Tensor biases_f32 = weights_biases_f32[-1];

    // We combine the batch and chunk size dimensions into the leading dimension for input into torch::addmm.
    const at::Tensor in_f32 =
            torch::rand({lstm_chunk_size * in_batch_size, layer_size}, torch::kFloat32);

    // The kernel takes float16 weights, and works generally in float16.
    const at::Tensor weights_biases_f16 = weights_biases_f32.to(torch::kFloat16);
    const at::Tensor in_f16 = in_f32.to(torch::kFloat16);

    // Prepare the input buffer for the Linear kernel.
    // reorder_inputs transforms the input in 3 ways:
    // 1) Rearranges input tiles in a fairly complex manner.
    // 2) Adds one time step of padding before and after the chunk time extents.
    // 3) Converts from float32 to float16.
    at::Tensor in_f16_reordered =
            torch::zeros({lstm_chunk_size + 3, in_batch_size, layer_size}, torch::kFloat16);
    launch_kernel(reorder_input_cps.get(), command_queue.get(),
                  {args_reorder.get(), mtl_for_tensor(in_f32), mtl_for_tensor(in_f16_reordered)},
                  {}, kernel_thread_groups, threads_per_thread_group);

    // CPU comparison calculation.
    const at::Tensor out_cpu_f32 = torch::addmm(biases_f32, in_f32, weights_f32);

    for (bool output_clamp : {false, true}) {
        const auto out_cpu_clamp_f32 = output_clamp ? out_cpu_f32.clamp(-5.f, 5.f) : out_cpu_f32;
        for (bool output_tanh : {false, true}) {
            const auto out_cpu_tanh_f32 =
                    output_tanh ? out_cpu_clamp_f32.tanh() : out_cpu_clamp_f32;
            for (bool output_as_byte : {false, true}) {
                // Byte output is only supported if at least one of [tanh, clamp] is enabled
                if (output_as_byte && !(output_tanh || output_clamp)) {
                    continue;
                }
                float output_scale =
                        (output_as_byte ? ((output_clamp && !output_tanh) ? (127.f / 5.f) : 127.f)
                                        : 1.f);
                const auto out_cpu =
                        output_as_byte
                                ? (output_scale * out_cpu_tanh_f32).to(torch::kI8).to(torch::kF32)
                                : out_cpu_tanh_f32;

                for (bool input_from_lstm : {false, true}) {
                    CATCH_DYNAMIC_SECTION("Metal linear layer " << output_clamp << output_tanh
                                                                << output_as_byte
                                                                << input_from_lstm) {
                        const NS::SharedPtr<MTL::ComputePipelineState> linear_cps = make_cps(
                                device.get(), input_from_lstm ? "linear_from_rev_lstm" : "linear",
                                {{"kLinearInSize", layer_size},
                                 {"kLinearOutSize", out_size},
                                 {"kLinearOutputScale", output_scale},
                                 {"kLinearOutputClamp", output_clamp},
                                 {"kLinearOutputTanh", output_tanh},
                                 {"kLinearOutputAsByte", output_as_byte}},
                                threads_per_thread_group);
                        CATCH_REQUIRE(linear_cps);

                        auto out_gpu_f32 = torch::zeros({lstm_chunk_size, in_batch_size, out_size},
                                                        torch::kF32);

                        const int out_batch_size = in_batch_size / batch_split;
                        const int32_t out_batch_tiles = out_batch_size / tile_size;

                        const int in_batch_tiles = in_batch_size / tile_size;
                        for (int in_batch_tile_offset = 0; in_batch_tile_offset < in_batch_tiles;
                             in_batch_tile_offset += out_batch_tiles) {
                            const std::vector<int32_t> args_linear_{
                                    in_batch_tiles, in_batch_tile_offset, out_batch_tiles,
                                    lstm_chunk_size};
                            const NS::SharedPtr<MTL::Buffer> args_linear =
                                    create_vec_buffer(device.get(), args_linear_);
                            CATCH_REQUIRE(args_linear);

                            auto out_dtype = output_as_byte ? torch::kI8 : torch::kF16;
                            auto out_gpu_partial = torch::zeros(
                                    {lstm_chunk_size, out_batch_size, out_size}, out_dtype);

                            launch_kernel(
                                    linear_cps.get(), command_queue.get(),
                                    {args_linear.get(),
                                     mtl_for_tensor(input_from_lstm ? in_f16_reordered : in_f16),
                                     mtl_for_tensor(weights_biases_f16),
                                     mtl_for_tensor(out_gpu_partial)},
                                    tg_buffer_lens, kernel_thread_groups, threads_per_thread_group);

                            int in_batch_offset = in_batch_tile_offset * tile_size;
                            out_gpu_f32.slice(1, in_batch_offset,
                                              in_batch_offset + out_batch_size) = out_gpu_partial;
                        }

                        // These tolerances are somewhat arbitary, but we must account for GPU calculations in float16
                        // versus CPU calculations in float32.
                        // (The CPU calculation can be done in float16, but is too slow.)
                        const float kRelTolerance = 0.1f;
                        const float kAbsTolerance =
                                output_as_byte ? (output_tanh ? 7.f : 2.f) : 0.08;
                        const float kMeanAbsDiffTolerance = output_as_byte ? 0.15f : 0.008f;

                        auto out_gpu_2d = out_gpu_f32.view({-1, out_size});
                        CATCH_CHECK(
                                torch::allclose(out_cpu, out_gpu_2d, kRelTolerance, kAbsTolerance));
                        CATCH_CHECK(MeanAbsDiff(out_cpu, out_gpu_2d) < kMeanAbsDiffTolerance);
                    }
                }
            }
        }
    }
}
