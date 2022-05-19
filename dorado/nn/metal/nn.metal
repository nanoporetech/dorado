#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 8;

static float sigmoid(float x) {
    return 1.f / (1.f + metal::exp(-x));
}

static float tanh_fast(float x) {
    return 2.f * sigmoid(2.f * x) - 1.f;
}

typedef float ftype_in;
typedef metal::simdgroup_float8x8 simdgroup_ftype_in8x8;
#if 0
typedef float ftype;
typedef metal::simdgroup_float8x8 simdgroup_ftype8x8;
#else
typedef half ftype;
typedef metal::simdgroup_half8x8 simdgroup_ftype8x8;
#endif

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

constant int NUM_TRANSITIONS = 5;

kernel void scan(
    device const ScanArgs* args,
    device const ftype_in *in,
    device ftype_in *out,
    device const int *idx1,
    device const int *idx2,
    KERNEL_INDEX_INPUTS)
{
    int T = args->T;
    int N = args->N;
    int C = args->C;
    int ts_states = C * NUM_TRANSITIONS;
    int dir = args->dir;
    int chunk = gid;

    device const ftype_in *chunk_in = in + chunk * ts_states;
    device ftype_in *chunk_out = out + chunk * (T+1) * C;
    device ftype_in *alpha_init = chunk_out + ((dir == -1) ? C * T : 0);
    for (int c = tid; c < C; ++c) {
        alpha_init[c] = 0;
    }
    for (int ts = 0; ts < T; ++ts) {
        threadgroup_barrier(mem_flags::mem_device);
        device const ftype_in *ts_in = chunk_in + N * ts_states * ((dir == -1) ? T - ts - 1 : ts);
        device ftype_in *ts_alpha_in = alpha_init + C * dir * ts;
        device ftype_in *ts_alpha_out = ts_alpha_in + C * dir;
        float max_val = -1e38f;
        float vals[NUM_TRANSITIONS];
        for (int i = 0; i < NUM_TRANSITIONS; ++i) {
            int state = tid * NUM_TRANSITIONS + i;
            vals[i] = ts_in[idx1[state]] + ts_alpha_in[idx2[state]];
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
    device const ScanArgs* args,
    device ftype_in *fwd_post,
    device const ftype_in *bwd,
    device ftype_in *post,
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

struct LstmArgs {
    int layer_size;
    int reverse;
    int chunk_tiles;
    int chunk_size;
    int linear_layer_size;
};


kernel void reorder_weights(
    device const LstmArgs* lstm,
    device const ftype_in* W,
    device const ftype_in* U,
    device const ftype_in* b,
    device ftype* weights_buf,
    KERNEL_INDEX_INPUTS)
{
    bool reverse = lstm->reverse;
    int layer_size = lstm->layer_size;
    int stride = layer_size * 4;
    device const ftype_in* inputs[3] = { reverse ? W : U, reverse ? U : W, b };
    int in_rows[3] = { layer_size, layer_size, 1 };

    for (int m = 0; m < 3; ++m) {
        for (int r = gid; r < in_rows[m]; r += threadgroups) {
            for (int c = tid; c < layer_size; c += threads) {
                for (int gate = 0; gate < 4; ++gate) {
                    weights_buf[(m * layer_size + r) * stride + c * 4 + gate] = ftype(inputs[m][r * stride + gate * layer_size + c]);
                }
            }
        }
    }
}

kernel void reorder_input(
    device const LstmArgs* args,
    device const ftype_in* in,
    device ftype* out,
    KERNEL_INDEX_INPUTS)
{
    threadgroup ftype bfr[MAX_LAYER_SIZE * TILE_SIZE];
    int layer_size = args->layer_size;
    int layer_tiles = layer_size / TILE_SIZE;
    int chunk_tiles = args->chunk_tiles;
    int chunk_size = args->chunk_size;
    for (int chunk_tile = gid; chunk_tile < chunk_tiles; chunk_tile += threadgroups) {
        // note: at timestep=-1 and timestep=chunk_size we do zero-padding in order to avoid having to deal with the edges differently
        for (int timestep = -1; timestep <= chunk_size; ++timestep) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int chunk = 0; chunk < TILE_SIZE; ++chunk) {
                for (int col = tid; col < layer_size; col += threads) {
                    int idx = (timestep * chunk_tiles * TILE_SIZE + (chunk_tile * TILE_SIZE + chunk)) * layer_size + col;
                    ftype val = (timestep >= 0 && timestep < chunk_size) ? ftype(in[idx]) : ftype(0);
                    bfr[chunk * MAX_LAYER_SIZE + col] = val;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int tile = sid; tile < layer_tiles; tile += simdgroups) {
                simdgroup_ftype8x8 A;
                simdgroup_load(A, bfr + tile * TILE_SIZE, MAX_LAYER_SIZE);
                simdgroup_store(A, out, chunk_tiles * TILE_SIZE, ulong2(chunk_tile * TILE_SIZE, (timestep + 1) * layer_size + tile * TILE_SIZE));
            }
        }
    }
}

kernel void reorder_output(
    device const LstmArgs* args,
    device const ftype* in,
    device ftype_in* out,
    KERNEL_INDEX_INPUTS)
{
    threadgroup ftype bfr[MAX_LAYER_SIZE * TILE_SIZE];
    int layer_size = args->layer_size;
    int layer_tiles = layer_size / TILE_SIZE;
    int chunk_tiles = args->chunk_tiles;
    int chunk_size = args->chunk_size;
    for (int chunk_tile = gid; chunk_tile < chunk_tiles; chunk_tile += threadgroups) {
        for (int timestep = 0; timestep < chunk_size; ++timestep) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int tile = sid; tile < layer_tiles; tile += simdgroups) {
                simdgroup_ftype8x8 A;
                simdgroup_load(A, in, chunk_tiles * TILE_SIZE, ulong2(chunk_tile * TILE_SIZE, (timestep + 1) * layer_size + tile * TILE_SIZE));
                simdgroup_store(A, bfr + tile * TILE_SIZE, MAX_LAYER_SIZE);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int chunk = 0; chunk < TILE_SIZE; ++chunk) {
                for (int col = tid; col < layer_size; col += threads) {
                    int idx = (timestep * chunk_tiles * TILE_SIZE + (chunk_tile * TILE_SIZE + chunk)) * layer_size + col;
                    out[idx] = ftype_in(bfr[chunk * MAX_LAYER_SIZE + col]);
                }
            }
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
kernel void conv(
    device const ConvArgs* args,
    device const ftype* in,
    device const ftype* weights,
    device ftype* out,
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
                out[chunk * chunk_size_out * out_size + ts * out_size + output_idx] = sum * sigmoid(sum);
            }
        }
    }
}
 */

kernel void conv1_simd_reorder_weights
(
    device const ConvArgs* args,
    device const ftype_in* weights_in,
    device ftype* weights_out
) {
    const int win_size = 5;
    const int out_size = 4;
    for (int col = 0; col < TILE_SIZE; ++col) {
        int in_col = col % out_size;
        for (int tile = 0; tile < 6; ++tile) {
            for (int row = 0; row < TILE_SIZE; ++row) {
                int in_row = row  + 4 - (col / 4) - (tile * 2);
                weights_out[(tile * TILE_SIZE + row) * TILE_SIZE + col] = (in_row >= 0 && in_row < win_size) ? weights_in[in_row * out_size + in_col] : ftype(0);
            }
        }
        weights_out[6 * TILE_SIZE * TILE_SIZE + col] = weights_in[win_size * out_size + in_col];
    }
}

kernel void conv2_simd_reorder_weights
(
    device const ConvArgs* args,
    device const ftype_in* weights_in,
    device ftype* weights_out
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

// Just type conversion
kernel void conv3_simd_reorder_weights
(
    device const ConvArgs* args,
    device const ftype_in* weights_in,
    device ftype* weights_out
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
    device const int* num_elems,
    device const ftype_in* in,
    device ftype* out,
    KERNEL_INDEX_INPUTS
) {
    for (int elem = gid * threads + tid; elem < *num_elems; elem += threadgroups * threads) {
        out[elem] = in[elem];
    }
}



#define CONV1_SIMD_GROUPS 16
[[max_total_threads_per_threadgroup(CONV1_SIMD_GROUPS * 32)]]
kernel void conv1_simd
(
    device const ConvArgs* args,
    device const ftype* in_buf,
    device const ftype* weights_buf,
    device ftype* out_buf,
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
                    simd_out_buf[sid][i][elem] = val * sigmoid(val);
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
                    simd_out_buf[sid][i][elem] = val * sigmoid(val);
                }
                simdgroup_load(A[i], simd_out_buf[sid][i], TILE_SIZE);
                simdgroup_store(A[i], out_buf, out_stride, ulong2(((iter + 1) * 4 + i) * TILE_SIZE, tile_row * TILE_SIZE));
            }
        }
    }
}
#undef CONV1_SIMD_GROUPS

#define CONV2_SIMD_GROUPS 16
[[max_total_threads_per_threadgroup(CONV2_SIMD_GROUPS * 32)]]
kernel void conv2_simd
(
    device const ConvArgs* args,
    device const ftype* in_buf,
    device const ftype* weights_buf,
    device ftype* out_buf,
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
                    simd_out_buf[sid][i][elem] = val * sigmoid(val);
                }
                simdgroup_load(A[i], simd_out_buf[sid][i], TILE_SIZE);
                simdgroup_store(A[i], out_buf, out_stride, ulong2((iter * 4 + i) * TILE_SIZE, tile_row * TILE_SIZE));
            }
        }
    }
}
#undef CONV2_SIMD_GROUPS

#define SIMD_TILES_M 6
#define SIMD_TILES_N 4
#define CONV3_SIMD_GROUPS 4

[[max_total_threads_per_threadgroup(CONV3_SIMD_GROUPS * 32)]]
kernel void conv3_simd
(
    device const ConvArgs* args,
    device const ftype* in_buf,
    device const ftype* weights_buf,
    device ftype* out_buf,
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
                    simd_out_buf[sid][0][elem] = val * sigmoid(val);
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
#undef SIMD_TILES_M
#undef SIMD_TILES_N


// Layer size 96
#define LSTM_LAYER_SIZE 96
#define LSTM_SIMD_GROUPS 16
#define LSTM_REVERSE 0
#define LSTM_KERNEL_SUFFIX _96_fwd_16
#include "nn.h"

#define LSTM_LAYER_SIZE 96
#define LSTM_SIMD_GROUPS 16
#define LSTM_REVERSE 1
#define LSTM_KERNEL_SUFFIX _96_rev_16
#include "nn.h"

// Layer size 128
#define LSTM_LAYER_SIZE 128
#define LSTM_SIMD_GROUPS 16
#define LSTM_REVERSE 0
#define LSTM_KERNEL_SUFFIX _128_fwd_16
#include "nn.h"

#define LSTM_LAYER_SIZE 128
#define LSTM_SIMD_GROUPS 16
#define LSTM_REVERSE 1
#define LSTM_KERNEL_SUFFIX _128_rev_16
#include "nn.h"

// Layer size 192
#define LSTM_LAYER_SIZE 192
#define LSTM_SIMD_GROUPS 12
#define LSTM_REVERSE 0
#define LSTM_KERNEL_SUFFIX _192_fwd_12
#include "nn.h"

#define LSTM_LAYER_SIZE 192
#define LSTM_SIMD_GROUPS 12
#define LSTM_REVERSE 1
#define LSTM_KERNEL_SUFFIX _192_rev_12
#include "nn.h"

// Layer size 256
#define LSTM_LAYER_SIZE 256
#define LSTM_SIMD_GROUPS 32
#define LSTM_REVERSE 0
#define LSTM_KERNEL_SUFFIX _256_fwd_32
#include "nn.h"

#define LSTM_LAYER_SIZE 256
#define LSTM_SIMD_GROUPS 32
#define LSTM_REVERSE 1
#define LSTM_KERNEL_SUFFIX _256_rev_32
#include "nn.h"

// Layer size 384
#define LSTM_LAYER_SIZE 384
#define LSTM_SIMD_GROUPS 24
#define LSTM_REVERSE 0
#define LSTM_KERNEL_SUFFIX _384_fwd_24
#include "nn.h"

#define LSTM_LAYER_SIZE 384
#define LSTM_SIMD_GROUPS 24
#define LSTM_REVERSE 1
#define LSTM_KERNEL_SUFFIX _384_rev_24
#include "nn.h"

// Layer size 512
#define LSTM_LAYER_SIZE 512
#define LSTM_SIMD_GROUPS 32
#define LSTM_REVERSE 0
#define LSTM_KERNEL_SUFFIX _512_fwd_32
#include "nn.h"

#define LSTM_LAYER_SIZE 512
#define LSTM_SIMD_GROUPS 32
#define LSTM_REVERSE 1
#define LSTM_KERNEL_SUFFIX _512_rev_32
#include "nn.h"
