#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 8;

// Values set via the FunctionConstantValues object passed in at MTL::Function
// creation time.
constant int kLstmLayerSize [[function_constant(0)]];
constant bool kLstmReversedInTime [[function_constant(1)]];
constant int kLinearLayerSize [[function_constant(2)]];
constant bool kClampConvOutput [[function_constant(3)]];

namespace {

inline float sigmoid(float x) {
    return 1.f / (1.f + metal::exp(-x));
}

inline float tanh_fast(float x) {
    return 2.f * sigmoid(2.f * x) - 1.f;
}

inline float conv_activation(float x) {
    // SiLU / swish activation.
    const float y = x * sigmoid(x);
    if (kClampConvOutput) {
        // Note: the lower bound is inoperative, since SiLU(x) has a min. of ~-0.28.
        return clamp(y, -0.5f, 3.5f);
    }
    return y;
}

}

// Precision of input activations and weights (before conversion).
typedef float ftype_in;

// Precision of layer processing.
#if 0
typedef float ftype;
typedef metal::simdgroup_float8x8 simdgroup_ftype8x8;
#else
typedef half ftype;
typedef metal::simdgroup_half8x8 simdgroup_ftype8x8;
#endif

// Precision of back guides and posterior probabilities.
// (Scores are int8_t.)
typedef float ftype_out;

#define MAX_LAYER_SIZE 512
#define KERNEL_INDEX_INPUTS \
    uint tid [[thread_index_in_threadgroup]], \
    uint gid [[threadgroup_position_in_grid]], \
    uint sid [[simdgroup_index_in_threadgroup]], \
    uint simdgroups [[simdgroups_per_threadgroup]], \
    uint threadgroups [[threadgroups_per_grid]], \
    uint threads [[threads_per_threadgroup]]

struct ScanArgs {
    int T;
    int N;
    int C;
    int dir;
};

kernel void scan(
    device const ScanArgs* const args,
    // Scores are supplied in int8 form, in the range [-127, 127], and must be mapped to [-5, 5] before use.
    device const int8_t* const scores_in,
    device ftype_out* const out,
    device const int* const idx1,
    device const int* const idx2,
    KERNEL_INDEX_INPUTS)
{
    constexpr int NUM_TRANSITIONS = 5;

    const int T = args->T;
    const int N = args->N;
    const int C = args->C;
    const int ts_states = C * NUM_TRANSITIONS;
    const int dir = args->dir;
    const int chunk = gid;

    device const int8_t* const chunk_in = scores_in + chunk * ts_states;
    device ftype_out* const chunk_out = out + chunk * (T+1) * C;
    device ftype_out* const alpha_init = chunk_out + ((dir == -1) ? C * T : 0);
    for (int c = tid; c < C; ++c) {
        alpha_init[c] = 0.0f;
    }
    for (int ts = 0; ts < T; ++ts) {
        threadgroup_barrier(mem_flags::mem_device);
        device const auto* const ts_in = chunk_in + N * ts_states * ((dir == -1) ? T - ts - 1 : ts);
        device ftype_out* const ts_alpha_in = alpha_init + C * dir * ts;
        device ftype_out* const ts_alpha_out = ts_alpha_in + C * dir;

        float max_val = -1e38f;
        float vals[NUM_TRANSITIONS];
        for (int i = 0; i < NUM_TRANSITIONS; ++i) {
            const int state = tid * NUM_TRANSITIONS + i;
            // Rescale the score from int8 to a float in the range [-5.0, 5.0].
            const auto kScoreScale = static_cast<float>(5.0 / 127.0);
            const auto score = static_cast<float>(ts_in[idx1[state]]) * kScoreScale;
            vals[i] = score + ts_alpha_in[idx2[state]];
            max_val = max(max_val, vals[i]);
        }
        float sum = 0.f;
        for (int i = 0; i < NUM_TRANSITIONS; ++i) {
            sum += exp(vals[i] - max_val);
        }
        ts_alpha_out[tid] = max_val + log(sum);
    }
}

kernel void add_softmax(
    device const ScanArgs* const args,
    device ftype_out* const fwd_post,
    device const ftype_out* const bwd,
    KERNEL_INDEX_INPUTS)
{
    int T = args->T + 1;
    int C = args->C;
    int chunk = gid;
    int simd_lane = tid & 31;

    for (int ts = sid; ts < T; ts += simdgroups) {
        int ts_idx = (chunk * T + ts) * C;
        float max_val = -1e38;
        for (int i = simd_lane; i < C; i += 32) {
            float val = fwd_post[ts_idx + i] + bwd[ts_idx + i];
            max_val = max(max_val, val);
            fwd_post[ts_idx + i] = val;
        }
        max_val = simd_max(max_val);
        float sum = 0;
        for (int i = simd_lane; i < C; i += 32) {
            float val = exp(fwd_post[ts_idx + i] - max_val);
            sum += val;
            fwd_post[ts_idx + i] = val;
        }
        sum = simd_sum(sum);
        float rcp_sum = 1.f / sum;
        for (int i = simd_lane; i < C; i += 32) {
            fwd_post[ts_idx + i] *= rcp_sum;
        }
    }
}

struct ConvArgs {
    int in_size;
    int win_size;
    int out_size;
    int stride;
    int pad;
    int chunk_size_in; // NOTE: multiple of stride!
    int num_chunks;
};

/*
// Generic convolution implementation that does not assume special weight
// ordering or particular input/output sizes.  Should work in place of
// specialised versions below.
kernel void conv(
    device const ConvArgs* const args,
    device const ftype* const in,
    device const ftype* const weights,
    device ftype* const out,
    KERNEL_INDEX_INPUTS)
{
    const int in_size = args->in_size;
    const int win_size = args->win_size;
    const int dp_size = in_size * win_size;
    const int out_size = args->out_size;
    const int stride = args->stride;
    const int pad = args->pad;
    const int chunk_size_in = args->chunk_size_in;
    const int chunk_size_out = chunk_size_in / stride;
    const int num_chunks = args->num_chunks;

    for (int chunk = gid * threads + tid; chunk < num_chunks; chunk += threadgroups * threads) {
        for (int ts = 0; ts < chunk_size_out; ++ts) {
            int in_pos_start = (ts * stride - pad) * in_size;
            for (int output_idx = 0; output_idx < out_size; ++output_idx) {
                ftype sum = weights[dp_size * out_size + output_idx]; // bias
                for (int dp_pos = 0; dp_pos < dp_size; ++dp_pos) {
                    int in_pos = in_pos_start + dp_pos;
                    if (in_pos >= 0 && in_pos < chunk_size_in * in_size) {
                        sum += in[chunk * chunk_size_in * in_size + in_pos] * weights[dp_pos * out_size + output_idx];
                    }
                }
                out[chunk * chunk_size_out * out_size + ts * out_size + output_idx] = conv_activation(sum);
            }
        }
    }
}
*/

// Rearranges/replicates weights for v3-type conv1, where output feature size is 4.
kernel void conv1_out4_simd_reorder_weights
(
    device const ConvArgs* const args,
    device const ftype_in* const weights_in,
    device ftype* const weights_out
) {
    const int win_size = 5;
    const int out_size = 4;
    for (int col = 0; col < TILE_SIZE; ++col) {
        int in_col = col % out_size;
        for (int tile = 0; tile < 6; ++tile) {
            for (int row = 0; row < TILE_SIZE; ++row) {
                int in_row = row + 4 - (col / 4) - (tile * 2);
                weights_out[(tile * TILE_SIZE + row) * TILE_SIZE + col] = (in_row >= 0 && in_row < win_size) ? weights_in[in_row * out_size + in_col] : ftype(0);
            }
        }
        weights_out[6 * TILE_SIZE * TILE_SIZE + col] = weights_in[win_size * out_size + in_col];
    }
}

// Rearranges/replicates weights for v4-type conv1, where output feature size is 4.
// Currently just type conversion
kernel void conv1_out16_simd_reorder_weights
(
    device const ConvArgs* const args,
    device const ftype_in* const weights_in,
    device ftype* const weights_out
) {
    const int cols = args->out_size;
    const int rows = args->in_size * args->win_size + 1;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            weights_out[row * cols + col] = weights_in[row * cols + col];
        }
    }
}

// Rearranges/replicates weights for v3-type conv2, where input feature size is 4.
kernel void conv2_in4_simd_reorder_weights
(
    device const ConvArgs* const args,
    device const ftype_in* const weights_in,
    device ftype* const weights_out
) {
    for (int col = 0; col < 16; ++col) {
        for (int row = 0; row < 29; ++row) {
            int in_row = row - 4;
            if (in_row >= 20) { in_row = -1; }
            if (row == 28) { in_row = 20; }
            weights_out[row * 16 + col] = (in_row >= 0) ? weights_in[in_row * 16 + col] : ftype(0);
        }
    }
}

// Rearranges/replicates weights for v4-type conv2, where input feature size is 16.
// Currently just type conversion
kernel void conv2_in16_simd_reorder_weights
(
    device const ConvArgs* const args,
    device const ftype_in* const weights_in,
    device ftype* const weights_out
) {
    const int cols = args->out_size;
    const int rows = args->in_size * args->win_size + 1;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            weights_out[row * cols + col] = weights_in[row * cols + col];
        }
    }
}

// Just type conversion
kernel void conv3_simd_reorder_weights
(
    device const ConvArgs* const args,
    device const ftype_in* const weights_in,
    device ftype* const weights_out
) {
    const int cols = args->out_size;
    const int rows = args->in_size * args->win_size + 1;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            weights_out[row * cols + col] = weights_in[row * cols + col];
        }
    }
}

// Just type conversion
kernel void float_to_half
(
    device const int* const num_elems,
    device const float* const in,
    device half* const out,
    KERNEL_INDEX_INPUTS
) {
    for (int elem = gid * threads + tid; elem < *num_elems; elem += threadgroups * threads) {
        out[elem] = in[elem];
    }
}

// Specialised conv1 implementation for v3-type models, where output feature size is 4.
#define CONV1_SIMD_GROUPS 16
[[max_total_threads_per_threadgroup(CONV1_SIMD_GROUPS * 32)]]
kernel void conv1_out4_simd
(
    device const ConvArgs* const args,
    device const ftype* const in_buf,
    device const ftype* const weights_buf,
    device ftype* const out_buf,
    KERNEL_INDEX_INPUTS
) {
//    const int in_size = 1;
    const int out_size = 4;
    const int chunk_size = args->chunk_size_in; // must be multiple of 8
    const int chunk_tiles = args->num_chunks / TILE_SIZE; // num_chunks must be multiple of TILE_SIZE
    const int out_stride = (chunk_size + 8) * out_size;
    threadgroup ftype simd_out_buf[CONV1_SIMD_GROUPS][4][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 W[6], B, I[2], A[4];

    for (int i = 0; i < 6; ++i) {
        simdgroup_load(W[i], weights_buf, TILE_SIZE, ulong2(0, i * TILE_SIZE));
    }
    simdgroup_load(B, weights_buf + 6 * TILE_SIZE * TILE_SIZE, 0);
    int num_iters = (chunk_size / 8) - 1;

    // Deal with the chunk edges first
    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        for (int pass = sid; pass < 2; ++pass) {
            simdgroup_load(I[0], in_buf, chunk_size, ulong2(pass * (chunk_size - 8), tile_row * TILE_SIZE));
            if (pass == 0) {
                A[0] = simdgroup_ftype8x8(0);
                simdgroup_multiply_accumulate(A[1], I[0], W[1], B);
                simdgroup_multiply_accumulate(A[2], I[0], W[2], B);
                simdgroup_multiply_accumulate(A[3], I[0], W[3], B);
            } else {
                simdgroup_multiply_accumulate(A[0], I[0], W[4], B);
                A[1] = simdgroup_ftype8x8(0);
                A[2] = simdgroup_ftype8x8(0);
                A[3] = simdgroup_ftype8x8(0);
            }
            for (int i = 0; i < 4; ++i) {
                simdgroup_store(A[i], simd_out_buf[sid][i], TILE_SIZE);
                for (int elem = tid & 31; elem < TILE_SIZE * TILE_SIZE; elem += 32) {
                    ftype val = simd_out_buf[sid][i][elem];
                    simd_out_buf[sid][i][elem] = conv_activation(val);
                }
                simdgroup_load(A[i], simd_out_buf[sid][i], TILE_SIZE);
                simdgroup_store(A[i], out_buf, out_stride, ulong2((pass * (num_iters + 1) * 4 + i) * TILE_SIZE, tile_row * TILE_SIZE));
            }
        }
    }

    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        for (int iter = sid; iter < num_iters; iter += simdgroups) {
            simdgroup_load(I[0], in_buf, chunk_size, ulong2(iter * TILE_SIZE, tile_row * TILE_SIZE));
            simdgroup_load(I[1], in_buf, chunk_size, ulong2((iter + 1) * TILE_SIZE, tile_row * TILE_SIZE));
            simdgroup_multiply_accumulate(A[0], I[0], W[4], B);
            simdgroup_multiply_accumulate(A[1], I[0], W[5], B);
            simdgroup_multiply_accumulate(A[2], I[1], W[2], B);
            simdgroup_multiply_accumulate(A[3], I[1], W[3], B);
            simdgroup_multiply_accumulate(A[0], I[1], W[0], A[0]);
            simdgroup_multiply_accumulate(A[1], I[1], W[1], A[1]);
            for (int i = 0; i < 4; ++i) {
                simdgroup_store(A[i], simd_out_buf[sid][i], TILE_SIZE);
                for (int elem = tid & 31; elem < TILE_SIZE * TILE_SIZE; elem += 32) {
                    ftype val = simd_out_buf[sid][i][elem];
                    simd_out_buf[sid][i][elem] = conv_activation(val);
                }
                simdgroup_load(A[i], simd_out_buf[sid][i], TILE_SIZE);
                simdgroup_store(A[i], out_buf, out_stride, ulong2(((iter + 1) * 4 + i) * TILE_SIZE, tile_row * TILE_SIZE));
            }
        }
    }
}

#undef CONV1_SIMD_GROUPS

// Conv1 implementation for v4-type models, where output feature size is 16.
// Currently this is just a copy of the generic implementation.
kernel void conv1_out16_simd
(
    device const ConvArgs* const args,
    device const ftype* const in,
    device const ftype* const weights,
    device ftype* const out,
    KERNEL_INDEX_INPUTS
) {
    const int in_size = args->in_size;
    const int win_size = args->win_size;
    const int dp_size = in_size * win_size;
    const int out_size = args->out_size;
    const int stride = args->stride;
    const int pad = args->pad;
    const int chunk_size_in = args->chunk_size_in;
    const int chunk_size_out = chunk_size_in / stride;
    const int num_chunks = args->num_chunks;

    for (int chunk = gid * threads + tid; chunk < num_chunks; chunk += threadgroups * threads) {
        for (int ts = 0; ts < chunk_size_out; ++ts) {
            int in_pos_start = (ts * stride - pad) * in_size;
            for (int output_idx = 0; output_idx < out_size; ++output_idx) {
                ftype sum = weights[dp_size * out_size + output_idx]; // bias
                for (int dp_pos = 0; dp_pos < dp_size; ++dp_pos) {
                    int in_pos = in_pos_start + dp_pos;
                    if (in_pos >= 0 && in_pos < chunk_size_in * in_size) {
                        sum += in[chunk * chunk_size_in * in_size + in_pos] * weights[dp_pos * out_size + output_idx];
                    }
                }
                out[chunk * chunk_size_out * out_size + ts * out_size + output_idx] = conv_activation(sum);
            }
        }
    }
}

// Specialised conv1 implementation for v3-type models, where input feature size is 4.
#define CONV2_SIMD_GROUPS 16
[[max_total_threads_per_threadgroup(CONV2_SIMD_GROUPS * 32)]]
kernel void conv2_in4_simd
(
    device const ConvArgs* const args,
    device const ftype* const in_buf,
    device const ftype* const weights_buf,
    device ftype* const out_buf,
    KERNEL_INDEX_INPUTS
) {
    const int in_size = 4;
    const int out_size = 16;
    const int chunk_size = args->chunk_size_in; // must be multiple of 2
    const int chunk_tiles = args->num_chunks / TILE_SIZE; // num_chunks must be multiple of TILE_SIZE
    const int in_stride = (chunk_size + 8) * in_size;
    const int out_stride = chunk_size * out_size;
    threadgroup ftype simd_out_buf[CONV2_SIMD_GROUPS][4][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 W[3][4], B[2], I[3], A[4];
    device const ftype* b = weights_buf + 28 * 16;

    simdgroup_load(B[0], b, 0);
    simdgroup_load(B[1], b + TILE_SIZE, 0);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            simdgroup_load(W[j][i * 2 + 0], weights_buf, 2 * TILE_SIZE, ulong2(0 * TILE_SIZE, j * TILE_SIZE + (1 - i) * 4));
            simdgroup_load(W[j][i * 2 + 1], weights_buf, 2 * TILE_SIZE, ulong2(1 * TILE_SIZE, j * TILE_SIZE + (1 - i) * 4));
        }
    }

    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        for (int iter = sid; iter < chunk_size / 2; iter += simdgroups) {
            simdgroup_load(I[0], in_buf, in_stride, ulong2((iter + 0) * TILE_SIZE, tile_row * TILE_SIZE));
            simdgroup_load(I[1], in_buf, in_stride, ulong2((iter + 1) * TILE_SIZE, tile_row * TILE_SIZE));
            simdgroup_load(I[2], in_buf, in_stride, ulong2((iter + 2) * TILE_SIZE, tile_row * TILE_SIZE));
            for (int i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(A[i], I[0], W[0][i], B[i & 1]);
                simdgroup_multiply_accumulate(A[i], I[1], W[1][i], A[i]);
                simdgroup_multiply_accumulate(A[i], I[2], W[2][i], A[i]);
            }
            for (int i = 0; i < 4; ++i) {
                simdgroup_store(A[i], simd_out_buf[sid][i], TILE_SIZE);
                for (int elem = tid & 31; elem < TILE_SIZE * TILE_SIZE; elem += 32) {
                    ftype val = simd_out_buf[sid][i][elem];
                    simd_out_buf[sid][i][elem] = conv_activation(val);
                }
                simdgroup_load(A[i], simd_out_buf[sid][i], TILE_SIZE);
                simdgroup_store(A[i], out_buf, out_stride, ulong2((iter * 4 + i) * TILE_SIZE, tile_row * TILE_SIZE));
            }
        }
    }
}
#undef CONV2_SIMD_GROUPS

// Conv2 implementation for v4-type models, where input feature size is 16.
// Currently this is just a copy of the generic implementation.
kernel void conv2_in16_simd
(
    device const ConvArgs* const args,
    device const ftype* const in,
    device const ftype* const weights,
    device ftype* const out,
    KERNEL_INDEX_INPUTS
) {
    const int in_size = args->in_size;
    const int win_size = args->win_size;
    const int dp_size = in_size * win_size;
    const int out_size = args->out_size;
    const int stride = args->stride;
    const int pad = args->pad;
    const int chunk_size_in = args->chunk_size_in;
    const int chunk_size_out = chunk_size_in / stride;
    const int num_chunks = args->num_chunks;

    for (int chunk = gid * threads + tid; chunk < num_chunks; chunk += threadgroups * threads) {
        for (int ts = 0; ts < chunk_size_out; ++ts) {
            int in_pos_start = (ts * stride - pad) * in_size;
            for (int output_idx = 0; output_idx < out_size; ++output_idx) {
                ftype sum = weights[dp_size * out_size + output_idx]; // bias
                for (int dp_pos = 0; dp_pos < dp_size; ++dp_pos) {
                    int in_pos = in_pos_start + dp_pos;
                    if (in_pos >= 0 && in_pos < chunk_size_in * in_size) {
                        sum += in[chunk * chunk_size_in * in_size + in_pos] * weights[dp_pos * out_size + output_idx];
                    }
                }
                out[chunk * chunk_size_out * out_size + ts * out_size + output_idx] = conv_activation(sum);
            }
        }
    }
}

#define SIMD_TILES_M 6
#define SIMD_TILES_N 4
#define CONV3_SIMD_GROUPS 4

[[max_total_threads_per_threadgroup(CONV3_SIMD_GROUPS * 32)]]
kernel void conv3_simd
(
    device const ConvArgs* const args,
    device const ftype* const in_buf,
    device const ftype* const weights_buf,
    device ftype* const out_buf,
    KERNEL_INDEX_INPUTS
) {
    const int in_size = args->in_size;
    const int in_size_tiles = args->in_size / TILE_SIZE;
    const int win_size = args->win_size;
    const int dp_size = in_size * win_size;
    const int out_size = args->out_size;
    const int stride = args->stride;
    const int pad = args->pad;
    const int chunk_size_in = args->chunk_size_in;
    const int chunk_size_out = chunk_size_in / stride;
    const int num_chunks = args->num_chunks;
    const int m_blks = num_chunks / (TILE_SIZE * SIMD_TILES_M);
    const int n_blks = out_size / (TILE_SIZE * SIMD_TILES_N);
    const int k_blks = dp_size / TILE_SIZE;
    threadgroup ftype simd_out_buf[CONV3_SIMD_GROUPS][SIMD_TILES_N * SIMD_TILES_M][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 A[SIMD_TILES_M], B[SIMD_TILES_N], C[SIMD_TILES_M * SIMD_TILES_N];
    device const ftype* b = weights_buf + dp_size * out_size;
    const int in_buf_stride = chunk_size_in * in_size;

    for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
        for (int chunk = tid; chunk < SIMD_TILES_M * TILE_SIZE; chunk += threads) {
            for (int i = 0; i < out_size; ++i) {
                int idx = i * num_chunks + m_blk * SIMD_TILES_M * TILE_SIZE + chunk;
                out_buf[idx] = 0;
                out_buf[idx + chunk_size_out * out_size] = 0;
            }
        }
    }

    for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
        for (int ts = 0; ts < chunk_size_out; ++ts) {
            int start_pos = ts * stride - pad;
            int start_pad_tiles = max(0, -start_pos) * in_size_tiles;
            int end_pad_tiles = max(0, start_pos + win_size - chunk_size_in) * in_size_tiles;
            device ftype* out = out_buf + (ts + 1) * num_chunks * out_size; // One timestep of padding as required by LSTM
            for (int n_blk = sid; n_blk < n_blks; n_blk += simdgroups) {
                for (int i = 0; i < SIMD_TILES_N; ++i) {
                    for (int j = 0; j < SIMD_TILES_M; ++j) {
                        // load bias into all accumulator tiles
                        simdgroup_load(C[j * SIMD_TILES_N + i], b + (n_blk * SIMD_TILES_N + i) * TILE_SIZE, 0);
                    }
                }
                for (int k_blk = start_pad_tiles; k_blk < k_blks - end_pad_tiles; ++k_blk) {
                    for (int i = 0; i < SIMD_TILES_N; ++i) {
                        simdgroup_load(B[i], weights_buf, out_size, ulong2((n_blk * SIMD_TILES_N + i) * TILE_SIZE, k_blk * TILE_SIZE));
                    }
#define LOAD_A(x) simdgroup_load(A[x], in_buf, in_buf_stride, ulong2((start_pos * in_size_tiles + k_blk) * TILE_SIZE, (m_blk * SIMD_TILES_M + x) * TILE_SIZE))
#define SMAC(x,y) simdgroup_multiply_accumulate(C[x * SIMD_TILES_N + y], A[x], B[y], C[x * SIMD_TILES_N + y])
                    LOAD_A(0);
                    LOAD_A(1);
                    SMAC(0,0); SMAC(0,1); SMAC(0,2); SMAC(0,3);
                    LOAD_A(2);
                    SMAC(1,0); SMAC(1,1); SMAC(1,2); SMAC(1,3);
                    LOAD_A(3);
                    SMAC(2,0); SMAC(2,1); SMAC(2,2); SMAC(2,3);
                    LOAD_A(4);
                    SMAC(3,0); SMAC(3,1); SMAC(3,2); SMAC(3,3);
                    LOAD_A(5);
                    SMAC(4,0); SMAC(4,1); SMAC(4,2); SMAC(4,3);
                    SMAC(5,0); SMAC(5,1); SMAC(5,2); SMAC(5,3);
#undef LOAD_A
#undef SMAC
                }

                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    for (int j = 0; j < SIMD_TILES_N; ++j) {
                        int tile_idx = i * SIMD_TILES_N + j;
                        simdgroup_store(C[tile_idx], simd_out_buf[sid][tile_idx], TILE_SIZE);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = tid & 31; elem < SIMD_TILES_N * SIMD_TILES_M * TILE_SIZE * TILE_SIZE; elem += 32) {
                    // swish activation
                    ftype val = simd_out_buf[sid][0][elem];
                    simd_out_buf[sid][0][elem] = conv_activation(val);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    uint out_col = (m_blk * SIMD_TILES_M + i) * TILE_SIZE;
                    for (int j = 0; j < SIMD_TILES_N; ++j) {
                        int tile_idx = i * SIMD_TILES_N + j;
                        uint out_row = (n_blk * SIMD_TILES_N + j) * TILE_SIZE;
                        simdgroup_load(A[0], simd_out_buf[sid][tile_idx], TILE_SIZE);
                        simdgroup_store(A[0], out, num_chunks, ulong2(out_col, out_row));
                    }
                }
            }
        }
    }
}
#undef CONV3_SIMD_GROUPS

kernel void reorder_lstm_weights(
    device const ftype_in* const W,
    device const ftype_in* const U,
    device const ftype_in* const b,
    device ftype* const weights_buf,
    KERNEL_INDEX_INPUTS)
{
    const int stride = kLstmLayerSize * 4;
    device const ftype_in* const inputs[3] = { U, W, b };
    const int in_rows[3] = { kLstmLayerSize, kLstmLayerSize, 1 };

    for (int m = 0; m < 3; ++m) {
        for (int r = gid; r < in_rows[m]; r += threadgroups) {
            for (int c = tid; c < kLstmLayerSize; c += threads) {
                for (int gate = 0; gate < 4; ++gate) {
                    weights_buf[(m * kLstmLayerSize + r) * stride + c * 4 + gate] = ftype(inputs[m][r * stride + gate * kLstmLayerSize + c]);
                }
            }
        }
    }
}

struct LstmArgs {
    int batch_tiles;
    int chunk_size;
};

kernel void reorder_input(
    device const LstmArgs* const args,
    device const ftype_in* const in,
    device ftype* const out,
    KERNEL_INDEX_INPUTS)
{
    threadgroup ftype bfr[MAX_LAYER_SIZE * TILE_SIZE];
    const int layer_tiles = kLstmLayerSize / TILE_SIZE;
    const int batch_tiles = args->batch_tiles;
    const int chunk_size = args->chunk_size;
    for (int batch_tile = gid; batch_tile < batch_tiles; batch_tile += threadgroups) {
        // note: at timestep=-1 and timestep=chunk_size we do zero-padding in order to avoid having to deal with the edges differently
        for (int timestep = -1; timestep <= chunk_size; ++timestep) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int chunk = 0; chunk < TILE_SIZE; ++chunk) {
                for (int col = tid; col < kLstmLayerSize; col += threads) {
                    const int idx = (timestep * batch_tiles * TILE_SIZE + (batch_tile * TILE_SIZE + chunk)) * kLstmLayerSize + col;
                    const ftype val = (timestep >= 0 && timestep < chunk_size) ? ftype(in[idx]) : ftype(0);
                    bfr[chunk * MAX_LAYER_SIZE + col] = val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int tile = sid; tile < layer_tiles; tile += simdgroups) {
                simdgroup_ftype8x8 A;
                simdgroup_load(A, bfr + tile * TILE_SIZE, MAX_LAYER_SIZE);
                simdgroup_store(A, out, batch_tiles * TILE_SIZE, ulong2(batch_tile * TILE_SIZE, (timestep + 1) * kLstmLayerSize + tile * TILE_SIZE));
            }
        }
    }
}

kernel void reorder_output(
    device const LstmArgs* const args,
    device const ftype* const in,
    device ftype_out* const out,
    KERNEL_INDEX_INPUTS)
{
    threadgroup ftype bfr[MAX_LAYER_SIZE * TILE_SIZE];
    const int layer_tiles = kLstmLayerSize / TILE_SIZE;
    const int batch_tiles = args->batch_tiles;
    const int chunk_size = args->chunk_size;
    for (int batch_tile = gid; batch_tile < batch_tiles; batch_tile += threadgroups) {
        for (int timestep = 0; timestep < chunk_size; ++timestep) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int tile = sid; tile < layer_tiles; tile += simdgroups) {
                simdgroup_ftype8x8 A;
                simdgroup_load(A, in, batch_tiles * TILE_SIZE, ulong2(batch_tile * TILE_SIZE, (timestep + 1) * kLstmLayerSize + tile * TILE_SIZE));
                simdgroup_store(A, bfr + tile * TILE_SIZE, MAX_LAYER_SIZE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int chunk = 0; chunk < TILE_SIZE; ++chunk) {
                for (int col = tid; col < kLstmLayerSize; col += threads) {
                    const int idx = (timestep * batch_tiles * TILE_SIZE + (batch_tile * TILE_SIZE + chunk)) * kLstmLayerSize + col;
                    out[idx] = ftype_out(bfr[chunk * MAX_LAYER_SIZE + col]);
                }
            }
        }
    }
}

// Note: max_total_threads_per_threadgroup is set via ComputePipelineDescriptor,
// rather than an attribute here, since it depends on the SIMD group count,
// which varies according to LSTM layer size.
kernel void lstm(
        device const LstmArgs* const args,
        device ftype* const in_out,
        device const ftype* const weights_buf,
        device ftype* const state_buf,
        device ftype* const temp_result_buf,
        // The sizes of these buffers are set via MTL::ComputeCommandEncoder.
        // They depend on the SIMD group count.
        threadgroup ftype (* const simd_res_buf)[2 * TILE_SIZE * TILE_SIZE],
        threadgroup ftype (* const simd_out_buf)[TILE_SIZE * TILE_SIZE],
        KERNEL_INDEX_INPUTS) {
    const int chunk_size = args->chunk_size;
    const int batch_tiles = args->batch_tiles;
    const int m_blks = batch_tiles / SIMD_TILES_M;
    const int n_blks = kLstmLayerSize * 4 / (TILE_SIZE * SIMD_TILES_N);
    const int k_blks = kLstmLayerSize * 2 / TILE_SIZE;
    const int inout_stride = batch_tiles * TILE_SIZE;
    const int W_stride = kLstmLayerSize * 4;
    simdgroup_ftype8x8 A[SIMD_TILES_M], B[SIMD_TILES_N], C[SIMD_TILES_M * SIMD_TILES_N];
    device const ftype* const b = weights_buf + 2 * kLstmLayerSize * W_stride;

    const uint t_idx = tid & 31;
    const uint col_bits = t_idx & 3;
    const uint row = t_idx >> 2;
    const uint rb_idx = t_idx * 4;

    for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
        for (int chunk = tid; chunk < SIMD_TILES_M * TILE_SIZE; chunk += threads) {
            for (int i = 0; i < kLstmLayerSize; ++i) {
                state_buf[i * batch_tiles * TILE_SIZE + m_blk * SIMD_TILES_M * TILE_SIZE + chunk] =
                        0;
            }
        }
    }

    for (int iter = 0; iter < chunk_size; ++iter) {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const int timestep_out = kLstmReversedInTime ? chunk_size - iter : iter + 1;
        const int timestep_in = kLstmReversedInTime ? timestep_out : timestep_out - 1;
        device const ftype* const in = in_out + timestep_in * inout_stride * kLstmLayerSize;
        device ftype* const out = in_out + timestep_out * inout_stride * kLstmLayerSize;
        for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
            for (int n_blk = sid; n_blk < n_blks; n_blk += simdgroups) {
                for (int i = 0; i < SIMD_TILES_N; ++i) {
                    for (int j = 0; j < SIMD_TILES_M; ++j) {
                        simdgroup_load(C[j * SIMD_TILES_N + i],
                                       b + (n_blk * SIMD_TILES_N + i) * TILE_SIZE, 0);
                    }
                }
                for (int k_blk = 0; k_blk < k_blks; ++k_blk) {
                    for (int i = 0; i < SIMD_TILES_N; ++i) {
                        simdgroup_load(
                                B[i], weights_buf, W_stride,
                                ulong2((n_blk * SIMD_TILES_N + i) * TILE_SIZE, k_blk * TILE_SIZE));
                    }
#define LOAD_A(x)                          \
    simdgroup_load(A[x], in, inout_stride, \
                   ulong2((m_blk * SIMD_TILES_M + x) * TILE_SIZE, k_blk * TILE_SIZE))
#define SMAC(x, y) \
    simdgroup_multiply_accumulate(C[x * SIMD_TILES_N + y], A[x], B[y], C[x * SIMD_TILES_N + y]);
                    LOAD_A(0);
                    LOAD_A(1);
                    SMAC(0, 0);
                    SMAC(0, 1);
                    SMAC(0, 2);
                    SMAC(0, 3);
                    LOAD_A(2);
                    SMAC(1, 0);
                    SMAC(1, 1);
                    SMAC(1, 2);
                    SMAC(1, 3);
                    LOAD_A(3);
                    SMAC(2, 0);
                    SMAC(2, 1);
                    SMAC(2, 2);
                    SMAC(2, 3);
                    LOAD_A(4);
                    SMAC(3, 0);
                    SMAC(3, 1);
                    SMAC(3, 2);
                    SMAC(3, 3);
                    LOAD_A(5);
                    SMAC(4, 0);
                    SMAC(4, 1);
                    SMAC(4, 2);
                    SMAC(4, 3);
                    SMAC(5, 0);
                    SMAC(5, 1);
                    SMAC(5, 2);
                    SMAC(5, 3);
#undef LOAD_A
#undef SMAC
                }
                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    const uint out_chunk_base = (m_blk * SIMD_TILES_M + i) * TILE_SIZE;
                    const uint chunk_idx = out_chunk_base + row;
                    for (int j = 0; j < SIMD_TILES_N; j += 2) {
                        simdgroup_store(C[i * SIMD_TILES_N + j + 0], simd_res_buf[sid],
                                        2 * TILE_SIZE);
                        simdgroup_store(C[i * SIMD_TILES_N + j + 1], simd_res_buf[sid] + TILE_SIZE,
                                        2 * TILE_SIZE);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        const uint col = j * 2 + col_bits;
                        const uint out_col = n_blk * SIMD_TILES_N * 2 + col;
                        const uint out_idx = out_col * inout_stride + chunk_idx;
                        const float g = tanh_fast(simd_res_buf[sid][rb_idx + 0]);
                        const float i = sigmoid(simd_res_buf[sid][rb_idx + 1]);
                        const float f = sigmoid(simd_res_buf[sid][rb_idx + 2]);
                        const float o = sigmoid(simd_res_buf[sid][rb_idx + 3]);
                        const float state = f * state_buf[out_idx] + i * g;
                        const float h = o * tanh_fast(state);
                        state_buf[out_idx] = state;
                        simd_out_buf[sid][row * TILE_SIZE + col] = h;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_load(A[0], simd_out_buf[sid], TILE_SIZE);
                    simdgroup_store(A[0],
                                    (n_blk < n_blks - int(simdgroups)) ? temp_result_buf : out,
                                    inout_stride, ulong2(out_chunk_base, n_blk * TILE_SIZE));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
            for (int n_blk = sid; n_blk < n_blks - int(simdgroups); n_blk += simdgroups) {
                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    uint out_chunk_base = (m_blk * SIMD_TILES_M + i) * TILE_SIZE;
                    simdgroup_load(A[0], temp_result_buf, inout_stride,
                                   ulong2(out_chunk_base, n_blk * TILE_SIZE));
                    simdgroup_store(A[0], out, inout_stride,
                                    ulong2(out_chunk_base, n_blk * TILE_SIZE));
                }
            }
        }
    }
}

struct LinearArgs {
    int in_batch_tiles;
    int in_batch_tile_offset;
    int out_batch_tiles;
    int chunk_size;
};

// TODO:
// 1) Make tanh optional.
// 2) Make output type switchable.
kernel void linear(
        device const LinearArgs* const args,
        device ftype* const in_buf,
        device const ftype* const weights_buf,
        device int8_t* const out_buf,
        // The size of this buffer is set via MTL::ComputeCommandEncoder.
        // It depends on the SIMD group count.
        threadgroup ftype (* const simd_out_buf)[TILE_SIZE * TILE_SIZE],
        KERNEL_INDEX_INPUTS) {
    const int chunk_size = args->chunk_size;
    const int in_batch_tiles = args->in_batch_tiles;
    const int in_batch_tile_offset = args->in_batch_tile_offset;
    const int out_batch_tiles = args->out_batch_tiles;
    const int m_blks = out_batch_tiles / SIMD_TILES_M;
    const int n_blks = kLinearLayerSize / (TILE_SIZE * SIMD_TILES_N);
    const int k_blks = kLstmLayerSize / TILE_SIZE;
    const int in_stride = in_batch_tiles * TILE_SIZE;
    const int W_stride = kLinearLayerSize;
    const int out_stride = kLinearLayerSize;
    simdgroup_ftype8x8 A[SIMD_TILES_M], B[SIMD_TILES_N], C[SIMD_TILES_M * SIMD_TILES_N];

    device const ftype* const b = weights_buf + kLstmLayerSize * W_stride;

    for (int ts = gid; ts < chunk_size; ts += threadgroups) {
        auto in = in_buf + in_batch_tile_offset * TILE_SIZE +
                  (ts + 1) * in_batch_tiles * TILE_SIZE * kLstmLayerSize;
        auto out = out_buf + ts * kLinearLayerSize * out_batch_tiles * TILE_SIZE;
        for (int m_blk = 0; m_blk < m_blks; ++m_blk) {
            for (int n_blk = sid; n_blk < n_blks; n_blk += simdgroups) {
                for (int i = 0; i < SIMD_TILES_N; ++i) {
                    for (int j = 0; j < SIMD_TILES_M; ++j) {
                        simdgroup_load(C[j * SIMD_TILES_N + i],
                                       b + (n_blk * SIMD_TILES_N + i) * TILE_SIZE, 0);
                    }
                }
                for (int k_blk = 0; k_blk < k_blks; ++k_blk) {
                    for (int i = 0; i < SIMD_TILES_N; ++i) {
                        simdgroup_load(
                                B[i], weights_buf, W_stride,
                                ulong2((n_blk * SIMD_TILES_N + i) * TILE_SIZE, k_blk * TILE_SIZE));
                    }
#define LOAD_A(x)                       \
    simdgroup_load(A[x], in, in_stride, \
                   ulong2((m_blk * SIMD_TILES_M + x) * TILE_SIZE, k_blk * TILE_SIZE))
#define SMAC(x, y) \
    simdgroup_multiply_accumulate(C[x * SIMD_TILES_N + y], A[x], B[y], C[x * SIMD_TILES_N + y]);
                    LOAD_A(0);
                    LOAD_A(1);
                    SMAC(0, 0);
                    SMAC(0, 1);
                    SMAC(0, 2);
                    SMAC(0, 3);
                    LOAD_A(2);
                    SMAC(1, 0);
                    SMAC(1, 1);
                    SMAC(1, 2);
                    SMAC(1, 3);
                    LOAD_A(3);
                    SMAC(2, 0);
                    SMAC(2, 1);
                    SMAC(2, 2);
                    SMAC(2, 3);
                    LOAD_A(4);
                    SMAC(3, 0);
                    SMAC(3, 1);
                    SMAC(3, 2);
                    SMAC(3, 3);
                    LOAD_A(5);
                    SMAC(4, 0);
                    SMAC(4, 1);
                    SMAC(4, 2);
                    SMAC(4, 3);
                    SMAC(5, 0);
                    SMAC(5, 1);
                    SMAC(5, 2);
                    SMAC(5, 3);
#undef LOAD_A
#undef SMAC
                }
                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    for (int j = 0; j < SIMD_TILES_N; ++j) {
                        // Store this 8x8 tile to threadgroup memory as ftype.
                        simdgroup_store(C[i * SIMD_TILES_N + j], simd_out_buf[sid], TILE_SIZE);
                        
                        const uint tile_i = (m_blk * SIMD_TILES_M + i) * TILE_SIZE;
                        const uint tile_j = (n_blk * SIMD_TILES_N + j) * TILE_SIZE;

                        // Apply tanh activation, scale to byte range, and store to the output
                        // buffer.
                        for (int elem = tid & 31; elem < TILE_SIZE * TILE_SIZE; elem += 32) {
                            const int in_tile_i = elem / TILE_SIZE;
                            const int in_tile_j = elem % TILE_SIZE;
                            out[(tile_i + in_tile_i) * out_stride + tile_j + in_tile_j] =
                                static_cast<int8_t>(tanh_fast(simd_out_buf[sid][elem]) * 127.0f);

                        }
                    }
                }
            }
        }
    }
}
