#include "torch_utils/tensor_utils.h"

#include "utils/dev_utils.h"
#include "utils/simd.h"

#include <torch/serialize.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <ostream>
#include <span>
#include <sstream>
#include <vector>

#ifndef NDEBUG
#include <spdlog/spdlog.h>
#endif

namespace dorado::utils {

namespace {

#if !ENABLE_NEON_IMPL  // We only need the SIMD implementation when we have Neon support.
IFUNC_DEFINE_DEFAULT(void, convert_f32_to_f16_impl)(c10::Half* const dest,
                                                    const float* const src,
                                                    std::size_t count) {
    auto src_tensor_f32 = at::from_blob(const_cast<float*>(src), {static_cast<int64_t>(count)});
    auto src_tensor_f16 = src_tensor_f32.to(at::ScalarType::Half);
    std::memcpy(dest, src_tensor_f16.data_ptr(), count * sizeof(c10::Half));
}
#endif  // ENABLE_NEON_IMPL

#if ENABLE_AVX2_IMPL || ENABLE_NEON_IMPL
#if ENABLE_AVX2_IMPL
IFUNC_DEFINE_OPTIMISED(void, convert_f32_to_f16_impl, AVX2_F16C)
#else
void convert_f32_to_f16_impl
#endif
(c10::Half* const dest, const float* const src, std::size_t count) {
    if (!count) {
        return;
    }

#if ENABLE_AVX2_IMPL
    // There seems to be no improvement by unrolling this (tested on pipelinedev).
    static constexpr size_t kUnrollFactor = 1;
#else
    // An unroll factor of 2 gives ~30% improvement on Apple Silicon.
    // Any higher unrolling shows no difference.
    static constexpr size_t kUnrollFactor = 2;
#endif

    // Outer unroll.
    static constexpr size_t kUnroll = simd::kFloatsPerRegister * kUnrollFactor;

    // Main vectorised loop.
    const auto* src_ptr = src;
    auto* dest_ptr = dest;
    for (size_t chunk_i = 0; chunk_i < count / kUnroll; ++chunk_i) {
        for (size_t unroll_i = 0; unroll_i < kUnrollFactor; ++unroll_i) {
            const simd::FloatRegister elems_f32 = simd_load_f32(src_ptr);
            const simd::HalfRegister elems_f16 = simd_convert_f32_f16(elems_f32);
            simd_store_f16(dest_ptr, elems_f16);
            src_ptr += simd::kFloatsPerRegister;
            dest_ptr += simd::kFloatsPerRegister;
        }
    }

    // Loop for final floats.
    // TODO -- probably nicer to use masked loads/stores.
    const size_t remaining_count = count % kUnroll;
    for (size_t i = 0; i < remaining_count; ++i) {
        const simd::FloatRegister elem_f32 = simd_load1_f32(src_ptr);
        const simd::HalfRegister elem_f16 = simd_convert_f32_f16(elem_f32);
        simd_store1_f16(dest_ptr, elem_f16);
        ++src_ptr;
        ++dest_ptr;
    }
}
#endif  // ENABLE_AVX2_IMPL || ENABLE_NEON_IMPL

IFUNC_DECLARE(convert_f32_to_f16_impl, AVX2_F16C);

#if !ENABLE_NEON_IMPL  // We only need the SIMD implementation when we have Neon support.
IFUNC_DEFINE_DEFAULT(void, shift_scale_tensor_i16_to_f16_inplace_impl)(at::Tensor& tensor,
                                                                       float shift,
                                                                       float scale) {
    tensor = tensor.to(at::ScalarType::Float).sub_(shift).div_(scale).to(at::ScalarType::Half);
}
#endif  // ENABLE_NEON_IMPL

#if ENABLE_AVX2_IMPL || ENABLE_NEON_IMPL
#if ENABLE_AVX2_IMPL
IFUNC_DEFINE_OPTIMISED(void, shift_scale_tensor_i16_to_f16_inplace_impl, AVX2_F16C)
#else
void shift_scale_tensor_i16_to_f16_inplace_impl
#endif
(at::Tensor& tensor, float shift, float scale) {
#if ENABLE_AVX2_IMPL
    // Unrolling a bit seems to give best performance (tested on a prom).
    constexpr std::size_t kUnrollFactor = 4;
#else
    // No unrolling seems to perform best on NEON.
    constexpr std::size_t kUnrollFactor = 1;
#endif

    constexpr std::size_t elem_size = 2;
    constexpr std::size_t elems_per_block = simd::kFloatsPerRegister * kUnrollFactor;

    std::int16_t* data = static_cast<std::int16_t*>(tensor.mutable_data_ptr());
    const std::size_t size = tensor.numel();

    // Process in blocks.
    const simd::FloatRegister shift_f32 = simd_load1_f32(&shift);
    const simd::FloatRegister scale_f32 = simd_load1_f32(&scale);
    for (std::size_t block = 0; block < size / elems_per_block; block++) {
        for (std::size_t unroll_i = 0; unroll_i < kUnrollFactor; unroll_i++) {
            const simd::Int16Register data_i16 = simd_load_i16(data);
            const simd::FloatRegister data_f32 = simd_convert_i16_f32(data_i16);
            const simd::FloatRegister scaled_f32 =
                    simd_div_f32(simd_sub_f32(data_f32, shift_f32), scale_f32);
            const simd::HalfRegister scaled_f16 = simd_convert_f32_f16(scaled_f32);
            simd_store_f16(data, scaled_f16);
            data += simd::kFloatsPerRegister;
        }
    }

    // Do the tail.
    std::int16_t block_in[elems_per_block];
    c10::Half block_out[elems_per_block];
    const std::size_t remaining = size % elems_per_block;
    std::memcpy(block_in, data, remaining * elem_size);
    for (std::size_t i = 0; i < remaining; i++) {
        const float val = static_cast<float>(block_in[i]);
        block_out[i] = (val - shift) / scale;
    }
    std::memcpy(data, block_out, remaining * elem_size);

    // The tensor should now be viewed as halfs.
    tensor = tensor.view(at::ScalarType::Half);
}
#endif  // ENABLE_AVX2_IMPL || ENABLE_NEON_IMPL

IFUNC_DECLARE(shift_scale_tensor_i16_to_f16_inplace_impl, AVX2_F16C);

}  // namespace

void serialise_tensor(const at::Tensor& t, const std::string& path) {
    auto bytes = torch::pickle_save(t);
    std::ofstream fout(path);
    fout.write(bytes.data(), bytes.size());
    fout.close();
}

std::vector<at::Tensor> load_tensors(const std::filesystem::path& dir,
                                     const std::vector<std::string>& tensors) {
    auto weights = std::vector<at::Tensor>();
    for (const auto& tensor : tensors) {
        auto path = dir / tensor;
        torch::load(weights, path.string());
    }

    return weights;
}

void dump_tensor([[maybe_unused]] const at::Tensor& t,
                 [[maybe_unused]] const std::string& name,
                 [[maybe_unused]] const int layer_no) {
#ifndef NDEBUG
    static const bool is_active = utils::get_dev_opt<bool>("dump_tensors", false);
    if (!is_active) {
        return;
    }

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);

    // Extract milliseconds
    auto millis =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // Format timestamp as YYYYMMDD_hhmmss_nnn (n = milliseconds)
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_t), "%Y%m%d_%H%M%S") << "_" << std::setw(3)
        << std::setfill('0') << millis.count();

    // Construct file path
    oss << "." << name << "." << layer_no << ".tensor";
    std::string path = oss.str();

    utils::serialise_tensor(t, path);
    spdlog::warn("Writing {} to: {}", utils::print_size(t, name), path);
#endif
}

at::Tensor quantile(const at::Tensor& t, const at::Tensor& q) {
    assert(q.dtype() == at::ScalarType::Float);

    auto tmp = t.clone();
    auto [qval, qidx] = q.sort();
    auto res = at::empty_like(q);

    auto start = tmp.data_ptr<float>();
    auto end = tmp.data_ptr<float>() + tmp.size(0);

    for (int i = 0; i < q.size(0); i++) {
        auto m = tmp.data_ptr<float>() +
                 static_cast<size_t>((tmp.size(0) - 1) * qval[i].item<float>());
        std::nth_element(start, m, end);
        res[qidx[i]] = *m;
        start = m;
    }

    return res;
}

at::Tensor quantile_counting(const at::Tensor& t, const at::Tensor& q) {
    assert(q.dtype() == at::ScalarType::Float);

    auto p = t.data_ptr<int16_t>();
    auto range_min = t.min().item<int16_t>();
    auto range_max = t.max().item<int16_t>();

    size_t size = t.size(0);

    std::vector<int> counts(range_max - range_min + 1, 0);
    for (size_t i = 0; i < size; ++i) {
        counts[p[i] - range_min]++;
    }
    std::partial_sum(counts.begin(), counts.end(), counts.begin());

    auto res = at::empty_like(q);

    for (size_t idx = 0; idx < size_t(q.numel()); idx++) {
        int threshold = int(q[idx].item<float>() * (size - 1));
        for (int i = 0; i < int(counts.size()); ++i) {
            if (counts[i] > threshold) {
                res[idx] = i + range_min;
                break;
            }
        }
    }

    return res;
}

// Multiversioned function dispatch doesn't work across the dorado_lib linking
// boundary.  Without this wrapper, AVX machines still only execute the default
// version.
void convert_f32_to_f16(c10::Half* const dest, const float* const src, std::size_t count) {
    return IFUNC_GET(convert_f32_to_f16_impl)(dest, src, count);
}

void shift_scale_tensor_i16_to_f16_inplace(at::Tensor& tensor, float shift, float scale) {
    assert(tensor.dtype() == at::ScalarType::Short);
    assert(tensor.is_contiguous());
    assert(tensor.element_size() == 2);
    IFUNC_GET(shift_scale_tensor_i16_to_f16_inplace_impl)(tensor, shift, scale);
}

template <typename T, size_t Dims>
static void flip_inplace_internal(at::Tensor& x, size_t dim) {
    TORCH_CHECK(x.dim() == Dims, "flip_inplace() dimension mismatch: ", Dims, " != ", x.dim());

    // Both CPU and MPS can access the data directly.
    if (x.device() == at::DeviceType::CPU || x.device() == at::DeviceType::MPS) {
        const auto make_array = [](const at::IntArrayRef& array) {
            TORCH_CHECK(array.size() == Dims);
            std::array<int64_t, Dims> out;
            for (size_t i = 0; i < Dims; i++) {
                out.at(i) = array.at(i);
            }
            return out;
        };

        // Sort them by stride for better cache usage.
        std::array<int64_t, Dims> sorted_sizes = make_array(x.sizes());
        std::array<int64_t, Dims> sorted_strides = make_array(x.strides());
        int sorted_dim = dim;
        // This is O(Dims^2), but Dims isn't going to be big.
        for (int64_t pos = 0; pos < std::ssize(sorted_strides); pos++) {
            const auto stride_iter = sorted_strides.begin() + pos;
            const auto size_iter = sorted_sizes.begin() + pos;
            // Find the smallest remaining stride.
            const auto min_idx = std::min_element(stride_iter, sorted_strides.end()) - stride_iter;
            // Move it into the current position.
            std::iter_swap(stride_iter, stride_iter + min_idx);
            std::iter_swap(size_iter, size_iter + min_idx);
            // Update the dimension that we're flipping on.
            if (sorted_dim == pos) {
                sorted_dim = pos + min_idx;
            } else if (sorted_dim == pos + min_idx) {
                sorted_dim = pos;
            }
        }

        auto* const data = x.mutable_data_ptr<T>();
        if (sorted_dim == 0) {
            // If the dimension we're flipping on is the innermost then we need
            // to do the left/right calculation in the inner loop.
            const int64_t inner_size = sorted_sizes.at(0);
            const int64_t inner_stride = sorted_strides.at(0);
            const std::span<const int64_t, Dims - 1> outer_sizes =
                    std::span{sorted_sizes}.template subspan<1, Dims - 1>();
            const std::span<const int64_t, Dims - 1> outer_strides =
                    std::span{sorted_strides}.template subspan<1, Dims - 1>();

            const auto swap_inner = [&](int64_t offset) {
                for (int64_t inner = 0; inner < inner_size / 2; inner++) {
                    // Recalculating left+right every time is 2x faster than
                    // incrementing/decrementing by stride each iteration.
                    // (Seen with tensors 10x10x10 to 500x500x500 on x86 with GCC).
                    auto* left = data + offset + inner * inner_stride;
                    auto* right = data + offset + (inner_size - 1 - inner) * inner_stride;
                    std::iter_swap(left, right);
                }
            };

            // TODO: can we do some voodoo to generalise this?
            if constexpr (Dims == 2) {
                for (int64_t outer0 = 0; outer0 < outer_sizes[0]; outer0++) {
                    const int64_t offset = outer0 * outer_strides[0];
                    swap_inner(offset);
                }
            } else if constexpr (Dims == 3) {
                for (int64_t outer1 = 0; outer1 < outer_sizes[1]; outer1++) {
                    for (int64_t outer0 = 0; outer0 < outer_sizes[0]; outer0++) {
                        const int64_t offset =
                                outer0 * outer_strides[0] + outer1 * outer_strides[1];
                        swap_inner(offset);
                    }
                }
            } else {
                static_assert(Dims != 0, "Unimplemented loop");
            }

        } else {
            // If the dimension we're flipping on isn't the innermost one then
            // it's more efficient to keep the inner loops as small as possible.
            const int64_t outer_size = sorted_sizes.at(sorted_dim);
            const int64_t outer_stride = sorted_strides.at(sorted_dim);

            // Extract the elements of an array, skipping the dim that we're fliping on.
            // eg, [1, 2, 3] + dim=1 --> [1, 3]
            const auto make_array_with_dim_skip =
                    [sorted_dim](const std::array<int64_t, Dims>& array) {
                        std::array<int64_t, Dims - 1> out{};
                        int64_t* ptr = out.data();
                        for (int64_t i = 0; i < sorted_dim; i++) {
                            *ptr++ = array.at(i);
                        }
                        for (size_t i = sorted_dim + 1; i < Dims; i++) {
                            *ptr++ = array.at(i);
                        }
                        return out;
                    };
            const std::array<int64_t, Dims - 1> inner_sizes =
                    make_array_with_dim_skip(sorted_sizes);
            const std::array<int64_t, Dims - 1> inner_strides =
                    make_array_with_dim_skip(sorted_strides);

            for (int64_t outer = 0; outer < outer_size / 2; outer++) {
                auto* const left = data + outer * outer_stride;
                auto* const right = data + (outer_size - outer - 1) * outer_stride;

                // TODO: can we do some voodoo to generalise this?
                if constexpr (Dims == 2) {
                    for (int64_t inner0 = 0; inner0 < inner_sizes[0]; inner0++) {
                        const int64_t offset = inner0 * inner_strides[0];
                        std::swap(left[offset], right[offset]);
                    }
                } else if constexpr (Dims == 3) {
                    for (int64_t inner1 = 0; inner1 < inner_sizes[1]; inner1++) {
                        for (int64_t inner0 = 0; inner0 < inner_sizes[0]; inner0++) {
                            const int64_t offset =
                                    inner0 * inner_strides[0] + inner1 * inner_strides[1];
                            std::swap(left[offset], right[offset]);
                        }
                    }
                } else {
                    static_assert(Dims != 0, "Unimplemented loop");
                }
            }
        }
    } else {
        // TODO(DOR-1773): CUDA kernel
        x = x.flip(dim);
    }
}
void flip_inplace(at::Tensor& x, size_t dim) {
    const auto dispatch = [&]<typename T>() {
        // An unsized specialisation (ie dims is a runtime constant) is about half as fast.
        const int64_t dims = x.dim();
        switch (dims) {
        case 2:
            flip_inplace_internal<T, 2>(x, dim);
            break;
        case 3:
            flip_inplace_internal<T, 3>(x, dim);
            break;
        default:
            throw std::runtime_error("Unhandled tensor size in flip_inplace()");
        }
    };

    const at::ScalarType type = x.scalar_type();
    switch (type) {
    case at::ScalarType::Float:
        dispatch.operator()<float>();
        break;
    case at::ScalarType::Half:
        dispatch.operator()<c10::Half>();
        break;
    default:
        throw std::runtime_error("Unhandled tensor type in flip_inplace()");
    }
}

void copy_tensor_elems(at::Tensor& dest_tensor,
                       std::size_t dest_offset,
                       const at::Tensor& src_tensor,
                       std::size_t src_offset,
                       std::size_t count) {
    assert(dest_tensor.is_contiguous());
    assert(src_tensor.is_contiguous());
    assert(dest_offset + count <= size_t(dest_tensor.numel()));
    assert(src_offset + count <= size_t(src_tensor.numel()));

    if (dest_tensor.dtype() == src_tensor.dtype()) {
        // No conversion.
        auto* const dest_ptr = reinterpret_cast<std::byte*>(dest_tensor.data_ptr());
        const auto* const src_ptr = reinterpret_cast<std::byte*>(src_tensor.data_ptr());
        const size_t elem_size = dest_tensor.element_size();
        std::memcpy(&dest_ptr[dest_offset * elem_size], &src_ptr[src_offset * elem_size],
                    count * elem_size);
    } else if (dest_tensor.dtype() == at::ScalarType::Half &&
               src_tensor.dtype() == at::ScalarType::Float) {
        // float32 -> float16 conversion.
        auto* const dest_ptr = dest_tensor.data_ptr<c10::Half>();
        const auto* const src_ptr = src_tensor.data_ptr<float>();
        convert_f32_to_f16(&dest_ptr[dest_offset], &src_ptr[src_offset], count);
    } else {
        // Slow fallback path for other conversions.
        using at::indexing::Slice;
        dest_tensor.flatten().index_put_(
                {Slice(dest_offset, dest_offset + count)},
                src_tensor.flatten().index({Slice(src_offset, src_offset + count)}));
    }
}

ScaledTensor quantize_tensor(const at::Tensor& t, int dim) {
    auto fp_range = t.abs().amax(dim);
    constexpr int levels = 256;
    auto quant_scale = ((levels / 2) / fp_range);
    auto quant_max = (levels / 2) - 1;
    auto t_quant = (t * quant_scale.unsqueeze(dim)).round().clip(-quant_max, quant_max);
    return ScaledTensor{t_quant.to(at::ScalarType::Char), quant_scale.to(at::ScalarType::Float)};
}

std::string print_size(const at::Tensor& t, const std::string& name) {
    std::string size;
    std::stringstream ss;
    ss << name << " tensor size: [";
    for (int i = 0; i < t.dim(); i++) {
        ss << t.size(i);
        if (i + 1 < t.dim()) {
            ss << ", ";
        }
    }
    ss << "] dtype: " << t.dtype();
    return ss.str();
}

void print_tensor_shape(std::ostream& os, const at::Tensor& tensor, const std::string& delimiter) {
    if (!tensor.defined()) {
        os << "undefined";
        return;
    }
    for (size_t i = 0; i < std::size(tensor.sizes()); ++i) {
        if (i > 0) {
            os << delimiter;
        }
        os << tensor.size(i);
    }
}

std::string tensor_shape_as_string(const at::Tensor& tensor) {
    std::ostringstream oss;
    print_tensor_shape(oss, tensor, ", ");
    return oss.str();
}

void save_tensor(const at::Tensor& tensor, const std::string& file_path) {
    const std::vector<char> pickled = torch::pickle_save(tensor);
    std::ofstream fout(file_path, std::ios::out | std::ios::binary);
    fout.write(std::data(pickled), std::size(pickled));
}

}  // namespace dorado::utils
