// TODO: turn these into parameters
#define SIMD_TILES_M 6
#define SIMD_TILES_N 4

#define CONCAT2(A, B) A##B
#define CONCAT(A, B) CONCAT2(A, B)

#if LSTM_LAYER_SIZE == 0
#undef LSTM_LAYER_SIZE
#define LSTM_LAYER_SIZE layer_size
#define LSTM_MAX_LAYER_SIZE 512
#else
#define LSTM_MAX_LAYER_SIZE LSTM_LAYER_SIZE
#endif

[[max_total_threads_per_threadgroup(LSTM_SIMD_GROUPS * 32)]] kernel void CONCAT(lstm_simd,
                                                                                LSTM_KERNEL_SUFFIX)(
        device const LstmArgs* args,
        device ftype* in_out,
        device const ftype* weights_buf,
        device ftype* state_buf,
        device ftype* temp_result_buf,
        KERNEL_INDEX_INPUTS) {
    int chunk_size = args->chunk_size;
    int chunk_tiles = args->chunk_tiles;
    int m_blks = chunk_tiles / SIMD_TILES_M;
    int n_blks = LSTM_LAYER_SIZE * 4 / (TILE_SIZE * SIMD_TILES_N);
    int k_blks = LSTM_LAYER_SIZE * 2 / TILE_SIZE;
    int inout_stride = chunk_tiles * TILE_SIZE;
    int W_stride = LSTM_LAYER_SIZE * 4;
    threadgroup ftype simd_res_buf[LSTM_SIMD_GROUPS][2 * TILE_SIZE * TILE_SIZE];
    threadgroup ftype simd_out_buf[LSTM_SIMD_GROUPS][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 A[SIMD_TILES_M], B[SIMD_TILES_N], C[SIMD_TILES_M * SIMD_TILES_N];
    device const ftype* b = weights_buf + 2 * LSTM_LAYER_SIZE * W_stride;

    uint t_idx = tid & 31;
    uint col_bits = t_idx & 3;
    uint row = t_idx >> 2;
    uint rb_idx = t_idx * 4;

    for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
        for (int chunk = tid; chunk < SIMD_TILES_M * TILE_SIZE; chunk += threads) {
            for (int i = 0; i < LSTM_LAYER_SIZE; ++i) {
                state_buf[i * chunk_tiles * TILE_SIZE + m_blk * SIMD_TILES_M * TILE_SIZE + chunk] =
                        0;
            }
        }
    }

    for (int iter = 0; iter < chunk_size; ++iter) {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        int timestep_out = LSTM_REVERSE ? chunk_size - iter : iter + 1;
        int timestep_in = LSTM_REVERSE ? timestep_out : timestep_out - 1;
        device const ftype* in = in_out + timestep_in * inout_stride * LSTM_LAYER_SIZE;
        device ftype* out = in_out + timestep_out * inout_stride * LSTM_LAYER_SIZE;
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
                    uint out_chunk_base = (m_blk * SIMD_TILES_M + i) * TILE_SIZE;
                    uint chunk_idx = out_chunk_base + row;
                    for (int j = 0; j < SIMD_TILES_N; j += 2) {
                        simdgroup_store(C[i * SIMD_TILES_N + j + 0], simd_res_buf[sid],
                                        2 * TILE_SIZE);
                        simdgroup_store(C[i * SIMD_TILES_N + j + 1], simd_res_buf[sid] + TILE_SIZE,
                                        2 * TILE_SIZE);
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        uint col = j * 2 + col_bits;
                        uint out_col = n_blk * SIMD_TILES_N * 2 + col;
                        uint out_idx = out_col * inout_stride + chunk_idx;
                        float g = tanh_fast(simd_res_buf[sid][rb_idx + 0]);
                        float i = sigmoid(simd_res_buf[sid][rb_idx + 1]);
                        float f = sigmoid(simd_res_buf[sid][rb_idx + 2]);
                        float o = sigmoid(simd_res_buf[sid][rb_idx + 3]);
                        float state = f * state_buf[out_idx] + i * g;
                        float h = o * tanh_fast(state);
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

#if LSTM_REVERSE == 0
[[max_total_threads_per_threadgroup(LSTM_SIMD_GROUPS * 32)]] kernel void CONCAT(linear_tanh_simd,
                                                                                LSTM_KERNEL_SUFFIX)(
        device const LstmArgs* args,
        device ftype* in_buf,
        device const ftype* weights_buf,
        device ftype_in* out_buf,
        KERNEL_INDEX_INPUTS) {
    int chunk_size = args->chunk_size;
    int chunk_tiles = args->chunk_tiles;
    int linear_layer_size = args->linear_layer_size;
    int m_blks = chunk_tiles / SIMD_TILES_M;
    int n_blks = linear_layer_size / (TILE_SIZE * SIMD_TILES_N);
    int k_blks = LSTM_LAYER_SIZE / TILE_SIZE;
    int in_stride = chunk_tiles * TILE_SIZE;
    int W_stride = linear_layer_size;
    int out_stride = linear_layer_size;
    threadgroup ftype simd_out_buf[LSTM_SIMD_GROUPS][TILE_SIZE * TILE_SIZE];
    threadgroup float simd_out_buf_f32[LSTM_SIMD_GROUPS][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 A[SIMD_TILES_M], B[SIMD_TILES_N], C[SIMD_TILES_M * SIMD_TILES_N];
    simdgroup_float8x8 out_tile;
    device const ftype* b = weights_buf + LSTM_LAYER_SIZE * W_stride;

    for (int ts = gid; ts < chunk_size; ts += threadgroups) {
        auto in = in_buf + (ts + 1) * chunk_tiles * TILE_SIZE * LSTM_LAYER_SIZE;
        auto out = out_buf + ts * linear_layer_size * chunk_tiles * TILE_SIZE;
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
                        simdgroup_store(C[i * SIMD_TILES_N + j], simd_out_buf[sid], TILE_SIZE);
                        for (int elem = tid & 31; elem < TILE_SIZE * TILE_SIZE; elem += 32) {
                            simd_out_buf_f32[sid][elem] = 5.f * tanh_fast(simd_out_buf[sid][elem]);
                        }
                        simdgroup_load(out_tile, simd_out_buf_f32[sid], TILE_SIZE);
                        simdgroup_store(out_tile, out, out_stride,
                                        ulong2((n_blk * SIMD_TILES_N + j) * TILE_SIZE,
                                               (m_blk * SIMD_TILES_M + i) * TILE_SIZE));
                    }
                }
            }
        }
    }
}
#endif  // LSTM_REVERSE == 0

#undef LSTM_KERNEL_SUFFIX
#undef LSTM_LAYER_SIZE
#undef LSTM_MAX_LAYER_SIZE
#undef LSTM_SIMD_GROUPS
#undef LSTM_REVERSE
#undef SIMD_TILES_M
#undef SIMD_TILES_N
