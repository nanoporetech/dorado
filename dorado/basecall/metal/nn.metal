#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 8;

// Values set via the FunctionConstantValues object passed in at MTL::Function
// creation time.
constant int kLstmLayerSize [[function_constant(0)]];
constant bool kLstmReversedInTime [[function_constant(1)]];
constant int kLinearInSize [[function_constant(2)]];
constant int kLinearOutSize [[function_constant(3)]];
constant bool kConvOutputClamp [[function_constant(4)]];
constant float kLinearOutputScale [[function_constant(5)]];
constant bool kLinearOutputClamp [[function_constant(6)]];
constant bool kLinearOutputTanh [[function_constant(7)]];
constant bool kLinearOutputAsByte [[function_constant(8)]];
constant bool kConvTanhActivation [[function_constant(9)]];

namespace {

inline float sigmoid(float x) { return 1.f / (1.f + metal::exp(-x)); }

inline float tanh_fast(float x) { return 2.f * sigmoid(2.f * x) - 1.f; }

inline float conv_activation(float x) {
    // tanh or SiLU / swish activation.
    const float y = kConvTanhActivation ? tanh_fast(x) : x * sigmoid(x);
    if (kConvOutputClamp) {
        // We only clamp in the case of SiLU/swish, which has a minimum of ~0.28.
        // Only an upper bound need be imposed.
        return min(y, 3.5f);
    }
    return y;
}

}  // namespace

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
#define KERNEL_INDEX_INPUTS                                                  \
    [[maybe_unused]] uint tid [[thread_index_in_threadgroup]],               \
            [[maybe_unused]] uint gid [[threadgroup_position_in_grid]],      \
            [[maybe_unused]] uint sid [[simdgroup_index_in_threadgroup]],    \
            [[maybe_unused]] uint simdgroups [[simdgroups_per_threadgroup]], \
            [[maybe_unused]] uint threadgroups [[threadgroups_per_grid]],    \
            [[maybe_unused]] uint threads [[threads_per_threadgroup]]

struct ScanArgs {
    int T;
    int N;
    int C;
};

// Scores must be rescaled from byte range to [-5.0, 5.0] before use in
// forward / backward scans.
float ScaleByteScore(int8_t byte_score) {
    constexpr auto kScoreScale = static_cast<float>(5.0 / 127.0);
    return kScoreScale * static_cast<float>(byte_score);
}

kernel void backward_scan(const device ScanArgs* const args,
                          const device int8_t* const scores_in,
                          device ftype_out* const out,
                          KERNEL_INDEX_INPUTS) {
    constexpr int kNumBases = 4;
    constexpr int kNumTransitions = kNumBases + 1;
    constexpr float kFixedStayScore = 2.0f;

    const int T = args->T;
    const int N = args->N;
    const int num_states = args->C;
    const int ts_states = num_states * kNumBases;
    const int chunk = gid;

    const device int8_t* const chunk_in = scores_in + chunk * ts_states;
    device ftype_out* const chunk_out = out + chunk * (T + 1) * num_states;
    device ftype_out* const alpha_init = chunk_out + num_states * T;
    for (int c = tid; c < num_states; c += threads) {
        alpha_init[c] = 0.0f;
    }
    for (int ts = 0; ts < T; ++ts) {
        threadgroup_barrier(mem_flags::mem_device);
        const device auto* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        device ftype_out* const ts_alpha_in = alpha_init - num_states * ts;
        device ftype_out* const ts_alpha_out = ts_alpha_in - num_states;

        const int state = tid;
        const int stay_state_idx = state;
        const int step_state_idx_a = (state * kNumBases) % num_states;
        const int step_trans_idx_a =
                step_state_idx_a * kNumBases + ((state * kNumBases) / num_states);

        float vals[kNumTransitions];
        float max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
        for (int base = 0; base < kNumBases; ++base) {
            vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                             ScaleByteScore(ts_in[step_trans_idx_a + base * kNumBases]);
            max_val = max(max_val, vals[base + 1]);
        }
        float sum = 0.0f;
        for (int i = 0; i < kNumTransitions; ++i) {
            sum += exp(vals[i] - max_val);
        }
        ts_alpha_out[tid] = max_val + log(sum);
    }
}

// Performs the forward scan, writing out posterior probabilities as it goes.
// Forward scan results exist only transiently in threadgroup memory.
kernel void forward_scan_add_softmax(const device ScanArgs* const args,
                                     const device int8_t* const scores_in,
                                     const device ftype_out* const bwd,
                                     device int16_t* const post_int16,
                                     KERNEL_INDEX_INPUTS) {
    constexpr int kNumBases = 4;
    constexpr int kNumTransitions = kNumBases + 1;
    constexpr float kFixedStayScore = 2.0f;

    const int T = args->T + 1;       // Time steps over which we iterate.
    const int N = args->N;           // Batch size.
    const int num_states = args->C;  // kmer state space size
    const int chunk = gid;           // Batch element index.
    const int kMsb = num_states / kNumBases;
    const int ts_states = num_states * kNumBases;

    // This batch element's scores.
    const device int8_t* const chunk_scores = scores_in + chunk * ts_states;

    // TG buffers used to reduce max/sum across SIMD groups.
    constexpr int kMaxSIMDGroups = 32;
    threadgroup float sg_max_vals[kMaxSIMDGroups], sg_sums[kMaxSIMDGroups];

    // Alternating forward guide buffers used for successive time steps.
    constexpr int kMaxStates = 1024;
    threadgroup float ts_fwd[2][kMaxStates];

    // The forward guide input for the first step is 0.
    ts_fwd[0][tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ts = 0; ts < T; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.

        // This time step's scores.
        const device auto* const ts_scores = chunk_scores + N * ts_states * ts;

        // Alternating TG buffer twiddling.
        const threadgroup auto* const ts_alpha_in = ts_fwd[ts & 1];
        threadgroup auto* const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the next time step's forward guide from this time step's scores
        // and forward guide.  It's written to threadgroup memory for use in the
        // next iteration.
        const int state = tid;
        const int stay_state_idx = state;
        const int step_state_idx_a = state / kNumBases;
        const int step_trans_idx_a = state * kNumBases;
        float vals[kNumTransitions];
        float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
        for (int base = 0; base < kNumBases; ++base) {
            vals[base + 1] = ts_alpha_in[step_state_idx_a + base * kMsb] +
                             ScaleByteScore(ts_scores[step_trans_idx_a + base]);
            fwd_max_val = max(fwd_max_val, vals[base + 1]);
        }
        float fwd_sum = 0.0f;
        for (int i = 0; i < kNumTransitions; ++i) {
            fwd_sum += exp(vals[i] - fwd_max_val);
        }
        ts_alpha_out[tid] = fwd_max_val + log(fwd_sum);

        // Load the forward guide value calculated in the last time step for use
        // in this time step's posterior probability calculation.
        const float fwd_val = ts_alpha_in[tid];

        // Calculate fwd/bwd guide product in log space.
        const int ts_idx = (chunk * T + ts) * num_states;
        const float val = fwd_val + bwd[ts_idx + tid];

        // Determine max across this SIMD group, and write the result to
        // a threadgroup array with an entry for each SIMD group.
        sg_max_vals[sid] = simd_max(val);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find the max across all SIMD groups, and hence all entries
        // for this time step.
        float max_val = sg_max_vals[0];
        for (uint i = 1; i < simdgroups; ++i) {
            max_val = max(max_val, sg_max_vals[i]);
        }

        // Determine the sum of the exponentiated shifted log probabilities
        // across this SIMD group, and write the result to a threadgroup array
        // with an entry for each SIMD group.
        const float exp_val = exp(val - max_val);
        sg_sums[sid] = simd_sum(exp_val);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find the sum across all SIMD groups, and hence all entries
        // for this time step.
        float sum = sg_sums[0];
        for (uint i = 1; i < simdgroups; ++i) {
            sum += sg_sums[i];
        }

        // Write out the posterior probability, scaled to int16 range.
        post_int16[ts_idx + tid] =
                static_cast<int16_t>(round(clamp(exp_val / sum, 0.0f, 1.0f) * 32767.0f));
    }
}

kernel void backward_scan_float(const device ScanArgs* const args,
                                const device half* const scores_in,
                                device ftype_out* const out,
                                KERNEL_INDEX_INPUTS) {
    constexpr int kNumBases = 4;
    constexpr int kNumTransitions = kNumBases + 1;
    constexpr float kFixedStayScore = 2.0f;

    const int T = args->T;
    const int N = args->N;
    const int num_states = args->C;
    const int ts_states = num_states * kNumBases;
    const int chunk = gid;

    const device half* const chunk_in = scores_in + chunk * ts_states;
    device ftype_out* const chunk_out = out + chunk * (T + 1) * num_states;
    device ftype_out* const alpha_init = chunk_out + num_states * T;
    for (int c = tid; c < num_states; c += threads) {
        alpha_init[c] = 0.0f;
    }
    for (int ts = 0; ts < T; ++ts) {
        threadgroup_barrier(mem_flags::mem_device);
        const device auto* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        device ftype_out* const ts_alpha_in = alpha_init - num_states * ts;
        device ftype_out* const ts_alpha_out = ts_alpha_in - num_states;

        const int state = tid;
        const int stay_state_idx = state;
        const int step_state_idx_a = (state * kNumBases) % num_states;
        const int step_trans_idx_a =
                step_state_idx_a * kNumBases + ((state * kNumBases) / num_states);

        float vals[kNumTransitions];
        float max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
        for (int base = 0; base < kNumBases; ++base) {
            vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                             ts_in[step_trans_idx_a + base * kNumBases];
            max_val = max(max_val, vals[base + 1]);
        }
        float sum = 0.0f;
        for (int i = 0; i < kNumTransitions; ++i) {
            sum += exp(vals[i] - max_val);
        }
        ts_alpha_out[tid] = max_val + log(sum);
    }
}

// Performs the forward scan, writing out posterior probabilities as it goes.
// Forward scan results exist only transiently in threadgroup memory.
kernel void forward_scan_add_softmax_float(const device ScanArgs* const args,
                                           const device half* const scores_in,
                                           const device ftype_out* const bwd,
                                           device ftype_out* const posts,
                                           KERNEL_INDEX_INPUTS) {
    constexpr int kNumBases = 4;
    constexpr int kNumTransitions = kNumBases + 1;
    constexpr float kFixedStayScore = 2.0f;

    const int T = args->T + 1;       // Time steps over which we iterate.
    const int N = args->N;           // Batch size.
    const int num_states = args->C;  // kmer state space size
    const int chunk = gid;           // Batch element index.
    const int kMsb = num_states / kNumBases;
    const int ts_states = num_states * kNumBases;

    // This batch element's scores.
    const device half* const chunk_scores = scores_in + chunk * ts_states;

    // TG buffers used to reduce max/sum across SIMD groups.
    constexpr int kMaxSIMDGroups = 32;
    threadgroup float sg_max_vals[kMaxSIMDGroups], sg_sums[kMaxSIMDGroups];

    // Alternating forward guide buffers used for successive time steps.
    constexpr int kMaxStates = 1024;
    threadgroup float ts_fwd[2][kMaxStates];

    // The forward guide input for the first step is 0.
    ts_fwd[0][tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ts = 0; ts < T; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.

        // This time step's scores.
        const device auto* const ts_scores = chunk_scores + N * ts_states * ts;

        // Alternating TG buffer twiddling.
        const threadgroup auto* const ts_alpha_in = ts_fwd[ts & 1];
        threadgroup auto* const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the next time step's forward guide from this time step's scores
        // and forward guide.  It's written to threadgroup memory for use in the
        // next iteration.
        const int state = tid;
        const int stay_state_idx = state;
        const int step_state_idx_a = state / kNumBases;
        const int step_trans_idx_a = state * kNumBases;
        float vals[kNumTransitions];
        float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
        for (int base = 0; base < kNumBases; ++base) {
            vals[base + 1] = ts_alpha_in[step_state_idx_a + base * kMsb] +
                             ts_scores[step_trans_idx_a + base];
            fwd_max_val = max(fwd_max_val, vals[base + 1]);
        }
        float fwd_sum = 0.0f;
        for (int i = 0; i < kNumTransitions; ++i) {
            fwd_sum += exp(vals[i] - fwd_max_val);
        }
        ts_alpha_out[tid] = fwd_max_val + log(fwd_sum);

        // Load the forward guide value calculated in the last time step for use
        // in this time step's posterior probability calculation.
        const float fwd_val = ts_alpha_in[tid];

        // Calculate fwd/bwd guide product in log space.
        const int ts_idx = (chunk * T + ts) * num_states;
        const float val = fwd_val + bwd[ts_idx + tid];

        // Determine max across this SIMD group, and write the result to
        // a threadgroup array with an entry for each SIMD group.
        sg_max_vals[sid] = simd_max(val);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find the max across all SIMD groups, and hence all entries
        // for this time step.
        float max_val = sg_max_vals[0];
        for (uint i = 1; i < simdgroups; ++i) {
            max_val = max(max_val, sg_max_vals[i]);
        }

        // Determine the sum of the exponentiated shifted log probabilities
        // across this SIMD group, and write the result to a threadgroup array
        // with an entry for each SIMD group.
        const float exp_val = exp(val - max_val);
        sg_sums[sid] = simd_sum(exp_val);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Find the sum across all SIMD groups, and hence all entries
        // for this time step.
        float sum = sg_sums[0];
        for (uint i = 1; i < simdgroups; ++i) {
            sum += sg_sums[i];
        }

        // Write out the posterior probability
        posts[ts_idx + tid] = static_cast<ftype_out>(exp_val / sum);
    }
}

struct ConvArgs {
    int in_size;
    int win_size;
    int out_size;
    int stride;
    int pad;
    int chunk_size_in;  // NOTE: multiple of stride!
    int num_chunks;     // Actually batch size
    // These are in strided (output) time steps.
    // Ignored except by conv3.
    int conv3_time_step_begin;  // Inclusive.
    int conv3_time_step_end;    // Exclusive
};

/*
// Generic convolution implementation that assumes that `weights` are provided as contiguous tensor of shape
// [W, Cin, Cout] (or [window_size, in_size, out_size]), `in` is a contiguous tensor of shape [N, Tin, Cin] (or
// [num_chunks, chunk_size_in, in_size]), and `out` is a contiguous tensor of shape [N, Tout, Cout] (or
// [num_chunks, chunk_size_in / stride, out_size]).
//
// Should work in place of specialised versions below, provided zero-padding requirements are taken into account, and
// the output of the last convolution layer is passed to `reorder_input` before it is processed by the `lstm` kernel.
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

struct RowMajor {
    static ulong inner(int /* r */, int c) { return c; }
    static ulong outer(int r, int /* c */) { return r; }
};
struct ColMajor {
    static ulong inner(int r, int /* c */) { return r; }
    static ulong outer(int /* r */, int c) { return c; }
};
// 2D matrix layouts using 8x8 tiles. Note that RowMajor/ColMajor apply to the order of tiles, *NOT* within tiles
// RC == RowMajor: layout RrCc, where: R = row / 8; r = row % 8; C = col / 8; c = col % 8
// RC == ColMajor: layout CrRc, where: R = row / 8; r = row % 8; C = col / 8; c = col % 8
template <typename RC, typename FTYPE_PTR = const device ftype*, typename FTYPE = ftype>
struct TileBlock {
    using simdgroup_tile = simdgroup_matrix<FTYPE, TILE_SIZE, TILE_SIZE>;
    FTYPE_PTR ptr;
    int stride;
    TileBlock(FTYPE_PTR ptr_, int stride_, int r, int c)
            : ptr(ptr_ + RC::outer(r, c) * stride_ + RC::inner(r, c)), stride(stride_) {}
    void load(thread simdgroup_tile& tile, int r_tile, int c_tile) {
        simdgroup_load(tile, ptr, stride,
                       ulong2(RC::inner(r_tile, c_tile) * TILE_SIZE,
                              RC::outer(r_tile, c_tile) * TILE_SIZE));
    }
    void store(const thread simdgroup_tile& tile, int r_tile, int c_tile) {
        simdgroup_store(tile, ptr, stride,
                        ulong2(RC::inner(r_tile, c_tile) * TILE_SIZE,
                               RC::outer(r_tile, c_tile) * TILE_SIZE));
    }
};

// TNC matrix layouts (i.e. time, batch, channel)
template <int SIMD_TILES_N, int SIMD_TILES_C, typename FTYPE = ftype>
struct MatLayoutRowMajor {
    using TileBlockConst = TileBlock<RowMajor, const device FTYPE*, FTYPE>;
    using TileBlock = TileBlock<RowMajor, device FTYPE*, FTYPE>;
    using ftype = FTYPE;
    static TileBlock
    tnc_block(device FTYPE* const ptr, int /* T */, int N, int C, int t, int n_blk, int c_blk) {
        return TileBlock(ptr, C, t * N + n_blk * SIMD_TILES_N * TILE_SIZE,
                         c_blk * SIMD_TILES_C * TILE_SIZE);
    }
    static TileBlockConst tnc_block(const device FTYPE* const ptr,
                                    int /* T */,
                                    int N,
                                    int C,
                                    int t,
                                    int n_blk,
                                    int c_blk) {
        return TileBlockConst(ptr, C, t * N + n_blk * SIMD_TILES_N * TILE_SIZE,
                              c_blk * SIMD_TILES_C * TILE_SIZE);
    }
    static void zero_initial_state(device FTYPE* const, int, int, int, uint, uint, uint, uint) {}
};

// The memory layout of LSTM input/output matrices matches a contiguous tensor of sizes [T+3, C/8, 8, N/8, 8],
// corresponding to [t+x, c/8, n%8, n/8, c%8] in TNC terms (time, batch, channel).
// x is either 1 or 2, depending on whether the buffer is output from a forward or a reverse LSTM layer.
// Thus a reverse layer shifts its output one forward, a forward layer shifts it one back.
// This works as long as we alternate reverse and forward layers.
// [0] and [T+2] are set to zero and act as the initial LSTM state.
//
//
// To illustrate, the output of a conv3/forward LSTM layer with T=10 may look like this ('?' denotes
// uninitialised/irrelevant data, A is data for t=0, J for t=9):
//
//    position:    0   1   2   3   4   5   6   7   8   9  10  11  12
//        data:    0   A   B   C   D   E   F   G   H   I   J   ?   0
//
// The reverse LSTM layer, in its iteration 0, loads inputs J(10) and 0(12), and produces a new output J',
// which it writes in the unused space between the two inputs (11). Iteration 1 loads inputs I(9) and the
// just-generated J'(11) to produce output I'(10), etc.
//
//        t_in:        0   1   2   3   4   5   6   7   8   9
// iteration 0:    0   A   B   C   D   E   F   G   H   I  [J]  ?  [0]
//                                                           \   /
//                 0   A   B   C   D   E   F   G   H   I   J  {J'} 0
//
// iteration 1:    0   A   B   C   D   E   F   G   H  [I]  J  [J'] 0
//                                                       \   /
//                 0   A   B   C   D   E   F   G   H   I  {I'} J'  0
//
//    [...]
//
// iteration 9:    0  [A]  B  [B'] C'  D'  E'  F'  G'  H'  I'  J'  0
//                       \   /
//                 0   A  {A'} B'  C'  D'  E'  F'  G'  H'  I'  J'  0
//       t_out:            0   1   2   3   4   5   6   7   8   9
//
//
// The forward LSTM layer then loads inputs 0(0) and A'(2) in its iteration 0, producing new output A",
// overwriting A.
//
//        t_in:            0   1   2   3   4   5   6   7   8   9
// iteration 0:   [0]  A  [A'] B'  C'  D'  E'  F'  G'  H'  I'  J'  0
//                   \   /
//                 0  {A"} A'  B'  C'  D'  E'  F'  G'  H'  I'  J'  0
//
// iteration 1:    0  [A"] A' [B'] C'  D'  E'  F'  G'  H'  I'  J'  0
//                       \   /
//                 0   A" {B"} B'  C'  D'  E'  F'  G'  H'  I'  J'  0
//
//    [...]
//
// iteration 9:    0   A"  B"  C"  D"  E"  F"  G"  H" [I"] I' [J'] 0
//                                                       \   /
//                 0   A"  B"  C"  D"  E"  F"  G"  H"  I" {J"} J'  0
//       t_out:        0   1   2   3   4   5   6   7   8   9
//
//
// With this method we can use the same buffer for both input and output instead of needing twice as
// much memory for separate input and output buffers. Because the output of each LSTM iteration is
// written between the two inputs, we don't overwrite any data that is currently being consumed.
//
// NO_OFFSET is used in the lstm kernel to access raw positions in the T dimension
enum LstmOutputOffset { NO_OFFSET = 0, FORWARD_LSTM_OUTPUT = 1, REVERSE_LSTM_OUTPUT = 2 };
template <int SIMD_TILES_N, int SIMD_TILES_C, LstmOutputOffset T_OFFSET, typename FTYPE = ftype>
struct MatLayoutLSTM {
    using TileBlockConst = TileBlock<ColMajor, const device FTYPE*, FTYPE>;
    using TileBlock = TileBlock<ColMajor, device FTYPE*, FTYPE>;
    using ftype = FTYPE;
    static TileBlock
    tnc_block(device FTYPE* const ptr, int /* T */, int N, int C, int t, int n_blk, int c_blk) {
        return TileBlock(ptr, N, n_blk * SIMD_TILES_N * TILE_SIZE,
                         (t + T_OFFSET) * C + c_blk * SIMD_TILES_C * TILE_SIZE);
    }
    static TileBlockConst tnc_block(const device FTYPE* const ptr,
                                    int /* T */,
                                    int N,
                                    int C,
                                    int t,
                                    int n_blk,
                                    int c_blk) {
        return TileBlockConst(ptr, N, n_blk * SIMD_TILES_N * TILE_SIZE,
                              (t + T_OFFSET) * C + c_blk * SIMD_TILES_C * TILE_SIZE);
    }

    // Zero-initialise the inital LSTM state at T-positions 0 (for forward) and T+2 (for reverse)
    static void zero_initial_state(device FTYPE* const ptr,
                                   int T,
                                   int N,
                                   int C,
                                   uint gid,
                                   uint threadgroups,
                                   uint sid,
                                   uint simdgroups) {
        auto A = make_filled_simdgroup_matrix<FTYPE, TILE_SIZE, TILE_SIZE>(0);
        int n_tiles = N / TILE_SIZE;
        int c_tiles = C / TILE_SIZE;
        TileBlock first(ptr, N, 0, 0);
        TileBlock last(ptr, N, 0, (T + 2) * C);
        for (int c_tile = gid; c_tile < c_tiles; c_tile += threadgroups) {
            for (int n_tile = sid; n_tile < n_tiles; n_tile += simdgroups) {
                first.store(A, n_tile, c_tile);
                last.store(A, n_tile, c_tile);
            }
        }
    }
};

template <int SIMD_TILES_M, int SIMD_TILES_N>
class MatMul {
    static_assert(SIMD_TILES_M <= 8, "SIMD_TILES_M must be <= 8");
    static_assert(SIMD_TILES_N <= 8, "SIMD_TILES_N must be <= 8");
    simdgroup_ftype8x8 A, B, accum[SIMD_TILES_M][SIMD_TILES_N];

public:
    // Matrix multiply-accumulate.
    // Performance note: loading the A and B matrix tiles every iteration is actually
    // faster than trying to prefetch or keep the values in registers.
    template <typename LAYOUT_A, typename LAYOUT_B, bool N_INNER = true>
    void mma(int k_tiles_begin, int k_tiles_end, thread LAYOUT_A& mat_a, thread LAYOUT_B& mat_b) {
        for (int k_tile = k_tiles_begin; k_tile < k_tiles_end; ++k_tile) {
#pragma unroll 8
            for (int mn_tile = 0; mn_tile < 8; ++mn_tile) {
                // (N_INNER == true) means we compute accum[mn_tile][0:SIMD_TILES_N] per mn_tile iteration
                // (N_INNER == false) means we compute accum[0:SIMD_TILES_M][mn_tile] per mn_tile iteration
                if (N_INNER) {
                    const int m_tile = mn_tile;
                    if (m_tile < SIMD_TILES_M) {
                        mat_a.load(A, m_tile, k_tile);
                        for (int n_tile = 0; n_tile < SIMD_TILES_N; ++n_tile) {
                            mat_b.load(B, k_tile, n_tile);
                            simdgroup_multiply_accumulate(acc(m_tile, n_tile), A, B,
                                                          acc(m_tile, n_tile));
                        }
                    }
                } else {
                    const int n_tile = mn_tile;
                    if (n_tile < SIMD_TILES_N) {
                        mat_b.load(B, k_tile, n_tile);
                        for (int m_tile = 0; m_tile < SIMD_TILES_M; ++m_tile) {
                            mat_a.load(A, m_tile, k_tile);
                            simdgroup_multiply_accumulate(acc(m_tile, n_tile), A, B,
                                                          acc(m_tile, n_tile));
                        }
                    }
                }
            }
        }
    }

    void load_bias(const device ftype* const bias, int col) {
        for (int i = 0; i < SIMD_TILES_N; ++i) {
            for (int j = 0; j < SIMD_TILES_M; ++j) {
                simdgroup_load(acc(j, i), bias + col + i * TILE_SIZE, 0);
            }
        }
    }

    thread simdgroup_ftype8x8& acc(int m_tile, int n_tile) { return accum[m_tile][n_tile]; }
};

// Apply conv_activation to a simdgroup matrix tile (using threadgroup memory) then store the tile in device memory.
void conv_activation_and_store(simdgroup_ftype8x8 A,
                               threadgroup ftype* simd_local_buf,
                               int tid,
                               device ftype* const out_buf,
                               int out_stride,
                               int tile_col,
                               int tile_row) {
    simdgroup_store(A, simd_local_buf, TILE_SIZE);
    for (int elem = tid & 31; elem < TILE_SIZE * TILE_SIZE; elem += 32) {
        ftype val = simd_local_buf[elem];
        simd_local_buf[elem] = conv_activation(val);
    }
    simdgroup_load(A, simd_local_buf, TILE_SIZE);
    simdgroup_store(A, out_buf, out_stride, ulong2(tile_col * TILE_SIZE, tile_row * TILE_SIZE));
}

// Specialised conv1 implementation for v3-type simplex models, where output feature size is 4.
// Given a contiguous input tensor of shape [batch_size, chunk_size, 1] this will fill
// a contiguous output tensor of shape [batch_size, chunk_size + 8, 4], where the actual output
// is located at output.slice(1, 2, chunk_size + 2) and values before and after that slice are
// set to zero (this padding is used by the second layer in order to avoid special handling
// of the edges).
#define SIMD_GROUPS 16
[[max_total_threads_per_threadgroup(SIMD_GROUPS * 32)]]
kernel void conv1_in1_out4_simd(const device ConvArgs* const args,
                                const device ftype* const in_buf,
                                const device ftype* const weights_buf,
                                device ftype* const out_buf,
                                KERNEL_INDEX_INPUTS) {
    //    const int in_size = 1;
    const int out_size = 4;
    const int chunk_size = args->chunk_size_in;  // must be multiple of 8
    const int chunk_tiles =
            args->num_chunks / TILE_SIZE;  // num_chunks must be multiple of TILE_SIZE
    const int out_stride = (chunk_size + 8) * out_size;
    threadgroup ftype simd_out_buf[SIMD_GROUPS][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 W[4], B, I[2], A[4];

    for (int i = 0; i < 4; ++i) {
        simdgroup_load(W[i], weights_buf, TILE_SIZE, ulong2(0, 6 - 2 * i));
    }
    simdgroup_load(B, weights_buf + 14 * TILE_SIZE, 0);
    int num_iters = (chunk_size / TILE_SIZE) - 1;

    // Deal with the chunk edges first
    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        if (sid < 2) {
            int is_last = sid;
            simdgroup_load(I[0], in_buf, chunk_size,
                           ulong2(is_last * (chunk_size - 8), tile_row * TILE_SIZE));
            if (!is_last) {
                // Start of time span / output feature row.
                // Padded with 1 tile = 8 entries.
                A[0] = simdgroup_ftype8x8(0);
                simdgroup_multiply_accumulate(A[1], I[0], W[0], B);
                simdgroup_multiply_accumulate(A[2], I[0], W[1], B);
                simdgroup_multiply_accumulate(A[3], I[0], W[2], B);
            } else {
                // End of time span / output feature row.
                // Padded with 3 tiles = 8 entries.
                simdgroup_multiply_accumulate(A[0], I[0], W[3], B);
                A[1] = simdgroup_ftype8x8(0);
                A[2] = simdgroup_ftype8x8(0);
                A[3] = simdgroup_ftype8x8(0);
            }
            for (int i = 0; i < 4; ++i) {
                conv_activation_and_store(A[i], simd_out_buf[sid], tid, out_buf, out_stride,
                                          is_last * (num_iters + 1) * 4 + i, tile_row);
            }
        }
    }

    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        for (int iter = sid; iter < num_iters; iter += simdgroups) {
            simdgroup_load(I[0], in_buf, chunk_size,
                           ulong2(iter * TILE_SIZE + 4, tile_row * TILE_SIZE));
            simdgroup_load(I[1], in_buf, chunk_size,
                           ulong2((iter + 1) * TILE_SIZE, tile_row * TILE_SIZE));
            simdgroup_multiply_accumulate(A[0], I[0], W[1], B);
            simdgroup_multiply_accumulate(A[1], I[0], W[2], B);
            simdgroup_multiply_accumulate(A[2], I[1], W[1], B);
            simdgroup_multiply_accumulate(A[3], I[1], W[2], B);
            for (int i = 0; i < 4; ++i) {
                conv_activation_and_store(A[i], simd_out_buf[sid], tid, out_buf, out_stride,
                                          (iter + 1) * 4 + i, tile_row);
            }
        }
    }
}

#undef SIMD_GROUPS

// Conv1 implementation for v4-type simplex models, where output feature size is 16.
#define SIMD_GROUPS 16
[[max_total_threads_per_threadgroup(SIMD_GROUPS * 32)]]
kernel void conv1_in1_out16_simd(const device ConvArgs* const args,
                                 const device ftype* const in_buf,
                                 const device ftype* const weights_buf,
                                 device ftype* const out_buf,
                                 KERNEL_INDEX_INPUTS) {
    //    const int in_size = 1;
    const int out_size = 16;
    const int sid_parity =
            sid & 1;  // pairs of simdgroups work together, generating 8-wide tiles each
    const int chunk_size = args->chunk_size_in;  // must be multiple of TILE_SIZE
    const int chunk_tiles =
            args->num_chunks / TILE_SIZE;  // num_chunks must be multiple of TILE_SIZE
    const int out_stride = chunk_size * out_size;
    const int num_iters = (chunk_size / 4) - 1;
    threadgroup ftype simd_out_buf[SIMD_GROUPS][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 W[4], B, I, A;

    simdgroup_load(B, weights_buf + 15 * out_size + sid_parity * TILE_SIZE, 0);

    // Deal with the chunk edges first
    int is_last = sid / 2;
    simdgroup_load(W[0], weights_buf, 2 * TILE_SIZE,
                   ulong2(sid_parity * TILE_SIZE, is_last ? 1 : 7));
    simdgroup_load(W[1], weights_buf, 2 * TILE_SIZE,
                   ulong2(sid_parity * TILE_SIZE, is_last ? 0 : 6));
    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        if (sid < 4) {
            simdgroup_load(I, in_buf, chunk_size,
                           ulong2(is_last * (chunk_size - TILE_SIZE), tile_row * TILE_SIZE));
            int tile_col = is_last * (chunk_size - 2) * (out_size / TILE_SIZE) + sid_parity;
            simdgroup_multiply_accumulate(A, I, W[0], B);
            conv_activation_and_store(A, simd_out_buf[sid], tid, out_buf, out_stride, tile_col,
                                      tile_row);
            tile_col += (out_size / TILE_SIZE);
            simdgroup_multiply_accumulate(A, I, W[1], B);
            conv_activation_and_store(A, simd_out_buf[sid], tid, out_buf, out_stride, tile_col,
                                      tile_row);
        }
    }

    for (int i = 0; i < 4; ++i) {
        simdgroup_load(W[i], weights_buf, 2 * TILE_SIZE, ulong2(sid_parity * TILE_SIZE, 5 - i));
    }

    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        for (int iter = sid / 2; iter < num_iters; iter += simdgroups / 2) {
            simdgroup_load(I, in_buf, chunk_size, ulong2(iter * 4, tile_row * TILE_SIZE));
            for (int i = 0; i < 4; ++i) {
                int tile_col = (iter * 4 + 2 + i) * (out_size / TILE_SIZE) + sid_parity;
                simdgroup_multiply_accumulate(A, I, W[i], B);
                conv_activation_and_store(A, simd_out_buf[sid], tid, out_buf, out_stride, tile_col,
                                          tile_row);
            }
        }
    }
}
#undef SIMD_GROUPS

// Conv1 implementation for stereo duplex models, where input feature size is 13 and
// output feature size is 16.
// FIXME - replace this slow generic implementation.
kernel void conv1_in13_out16_simd(const device ConvArgs* const args,
                                  const device ftype* const in,
                                  const device ftype* const weights,
                                  device ftype* const out,
                                  KERNEL_INDEX_INPUTS) {
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
                ftype sum = weights[dp_size * out_size + output_idx];  // bias
                for (int dp_pos = 0; dp_pos < dp_size; ++dp_pos) {
                    int in_pos = in_pos_start + dp_pos;
                    if (in_pos >= 0 && in_pos < chunk_size_in * in_size) {
                        sum += in[chunk * chunk_size_in * in_size + in_pos] *
                               weights[dp_pos * out_size + output_idx];
                    }
                }
                out[chunk * chunk_size_out * out_size + ts * out_size + output_idx] =
                        conv_activation(sum);
            }
        }
    }
}

// Specialised conv2 implementation for v3-type models, where input feature size is 4.
#define SIMD_GROUPS 16
[[max_total_threads_per_threadgroup(SIMD_GROUPS * 32)]]
kernel void conv2_in4_out16_simd(const device ConvArgs* const args,
                                 const device ftype* const in_buf,
                                 const device ftype* const weights_buf,
                                 device ftype* const out_buf,
                                 KERNEL_INDEX_INPUTS) {
    const int in_size = 4;
    const int out_size = 16;
    const int chunk_size = args->chunk_size_in;  // must be multiple of 2
    const int chunk_tiles =
            args->num_chunks / TILE_SIZE;  // num_chunks must be multiple of TILE_SIZE
    const int in_stride = (chunk_size + 8) * in_size;
    const int out_stride = chunk_size * out_size;
    threadgroup ftype simd_out_buf[SIMD_GROUPS][TILE_SIZE * TILE_SIZE];
    simdgroup_ftype8x8 W[3][4], B[2], I[3], A[4];
    const device ftype* b = weights_buf + 28 * 16;

    simdgroup_load(B[0], b, 0);
    simdgroup_load(B[1], b + TILE_SIZE, 0);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            simdgroup_load(W[j][i * 2 + 0], weights_buf, 2 * TILE_SIZE,
                           ulong2(0 * TILE_SIZE, j * TILE_SIZE + (1 - i) * 4));
            simdgroup_load(W[j][i * 2 + 1], weights_buf, 2 * TILE_SIZE,
                           ulong2(1 * TILE_SIZE, j * TILE_SIZE + (1 - i) * 4));
        }
    }

    for (int tile_row = gid; tile_row < chunk_tiles; tile_row += threadgroups) {
        for (int iter = sid; iter < chunk_size / 2; iter += simdgroups) {
            simdgroup_load(I[0], in_buf, in_stride,
                           ulong2((iter + 0) * TILE_SIZE, tile_row * TILE_SIZE));
            simdgroup_load(I[1], in_buf, in_stride,
                           ulong2((iter + 1) * TILE_SIZE, tile_row * TILE_SIZE));
            simdgroup_load(I[2], in_buf, in_stride,
                           ulong2((iter + 2) * TILE_SIZE, tile_row * TILE_SIZE));
            for (int i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(A[i], I[0], W[0][i], B[i & 1]);
                simdgroup_multiply_accumulate(A[i], I[1], W[1][i], A[i]);
                simdgroup_multiply_accumulate(A[i], I[2], W[2][i], A[i]);
            }
            for (int i = 0; i < 4; ++i) {
                conv_activation_and_store(A[i], simd_out_buf[sid], tid, out_buf, out_stride,
                                          iter * 4 + i, tile_row);
            }
        }
    }
}
#undef SIMD_GROUPS

#define SIMD_TILES_M 6
#define SIMD_TILES_N 2
#define SIMD_GROUPS 4

[[max_total_threads_per_threadgroup(SIMD_GROUPS * 32)]]
kernel void conv2_in16_out16_simd(const device ConvArgs* const args,
                                  const device ftype* const in_buf,
                                  const device ftype* const weights_buf,
                                  device ftype* const out_buf,
                                  KERNEL_INDEX_INPUTS) {
    const int in_size = args->in_size;
    const int win_size = args->win_size;
    const int dp_size = in_size * win_size;
    const int out_size = 16;  // required! //args->out_size;
    const int stride = args->stride;
    const int pad = args->pad;
    const int w_pad_rows = 0;
    const int chunk_size_in = args->chunk_size_in;
    const int chunk_size_out = chunk_size_in / stride;
    const int num_chunks = args->num_chunks;
    const int m_blks = num_chunks / (TILE_SIZE * SIMD_TILES_M);
    const int k_tiles = (dp_size + TILE_SIZE - 1) / TILE_SIZE;
    threadgroup ftype simd_out_buf[SIMD_GROUPS][SIMD_TILES_N * SIMD_TILES_M][TILE_SIZE * TILE_SIZE];
    const device ftype* bias = weights_buf + (dp_size + 2 * w_pad_rows) * out_size;
    const int in_buf_stride = chunk_size_in * in_size;
    const int out_buf_stride = chunk_size_out * out_size;
    MatMul<SIMD_TILES_M, SIMD_TILES_N> mm;

    for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
        for (int ts = sid; ts < chunk_size_in / stride; ts += simdgroups) {
            int start_pos = (ts * stride - pad) * in_size;
            int end_pos = start_pos + dp_size;
            int clamped_start_pos = max(0, start_pos);
            int clamped_end_pos = min(chunk_size_in * in_size, end_pos);
            int start_pad = clamped_start_pos - start_pos;
            int end_pad = end_pos - clamped_end_pos;
            int start_pad_tiles = start_pad / TILE_SIZE;
            start_pad -= start_pad_tiles * TILE_SIZE;
            int w_row_offset = w_pad_rows - start_pad;
            int end_pad_tiles = end_pad / TILE_SIZE;
            int pad_tiles = start_pad_tiles + end_pad_tiles;
            mm.load_bias(bias, 0);
            TileBlock<RowMajor> mat_a(in_buf, in_buf_stride, m_blk * SIMD_TILES_M * TILE_SIZE,
                                      clamped_start_pos);
            TileBlock<RowMajor> mat_b(weights_buf, out_size, w_row_offset, 0);
            mm.mma(0, k_tiles - pad_tiles, mat_a, mat_b);
            for (int i = 0; i < SIMD_TILES_M; ++i) {
                for (int j = 0; j < SIMD_TILES_N; ++j) {
                    simdgroup_store(mm.acc(i, j), simd_out_buf[sid][i * SIMD_TILES_N + j],
                                    TILE_SIZE);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint elem = tid & 31; elem < SIMD_TILES_N * SIMD_TILES_M * TILE_SIZE * TILE_SIZE;
                 elem += 32) {
                // swish activation
                ftype val = simd_out_buf[sid][0][elem];
                simd_out_buf[sid][0][elem] = conv_activation(val);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = 0; i < SIMD_TILES_M; ++i) {
                int out_row = (m_blk * SIMD_TILES_M + i) * TILE_SIZE;
                for (int j = 0; j < SIMD_TILES_N; ++j) {
                    int tile_idx = i * SIMD_TILES_N + j;
                    int out_col = ts * out_size + j * TILE_SIZE;
                    simdgroup_ftype8x8 A;
                    simdgroup_load(A, simd_out_buf[sid][tile_idx], TILE_SIZE);
                    simdgroup_store(A, out_buf, out_buf_stride, ulong2(out_col, out_row));
                }
            }
        }
    }
}
#undef SIMD_GROUPS
#undef SIMD_TILES_M
#undef SIMD_TILES_N

#define SIMD_TILES_M 6
#define SIMD_TILES_N 4
#define SIMD_GROUPS 4

[[max_total_threads_per_threadgroup(SIMD_GROUPS * 32)]]
kernel void conv3_simd(const device ConvArgs* const args,
                       const device ftype* const in_buf,
                       const device ftype* const weights_buf,
                       device ftype* const out_buf,
                       KERNEL_INDEX_INPUTS) {
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
    // These are in terms of output time steps.
    const int time_step_begin = args->conv3_time_step_begin;
    const int time_step_end = args->conv3_time_step_end;
    const int m_blks = num_chunks / (TILE_SIZE * SIMD_TILES_M);
    const int n_blks = out_size / (TILE_SIZE * SIMD_TILES_N);
    const int k_blks = dp_size / TILE_SIZE;
    threadgroup ftype simd_out_buf[SIMD_GROUPS][SIMD_TILES_N * SIMD_TILES_M][TILE_SIZE * TILE_SIZE];
    const device ftype* bias = weights_buf + dp_size * out_size;
    const int in_buf_stride = chunk_size_in * in_size;
    MatMul<SIMD_TILES_M, SIMD_TILES_N> mm;
    using MatLayoutLSTM = MatLayoutLSTM<SIMD_TILES_M, SIMD_TILES_N, FORWARD_LSTM_OUTPUT>;

    MatLayoutLSTM::zero_initial_state(out_buf, chunk_size_out, num_chunks, out_size, gid,
                                      threadgroups, sid, simdgroups);

    for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
        for (int ts = time_step_begin; ts < time_step_end; ++ts) {
            int start_pos = ts * stride - pad;
            int start_pad_tiles = max(0, -start_pos) * in_size_tiles;
            int end_pad_tiles = max(0, start_pos + win_size - chunk_size_in) * in_size_tiles;
            for (int n_blk = sid; n_blk < n_blks; n_blk += simdgroups) {
                mm.load_bias(bias, n_blk * SIMD_TILES_N * TILE_SIZE);
                TileBlock<RowMajor> mat_a(in_buf, in_buf_stride, m_blk * SIMD_TILES_M * TILE_SIZE,
                                          start_pos * in_size_tiles * TILE_SIZE);
                TileBlock<RowMajor> mat_b(weights_buf, out_size, 0,
                                          n_blk * SIMD_TILES_N * TILE_SIZE);
                mm.mma(start_pad_tiles, k_blks - end_pad_tiles, mat_a, mat_b);
                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    for (int j = 0; j < SIMD_TILES_N; ++j) {
                        simdgroup_store(mm.acc(i, j), simd_out_buf[sid][i * SIMD_TILES_N + j],
                                        TILE_SIZE);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint elem = tid & 31;
                     elem < SIMD_TILES_N * SIMD_TILES_M * TILE_SIZE * TILE_SIZE; elem += 32) {
                    // swish activation
                    ftype val = simd_out_buf[sid][0][elem];
                    simd_out_buf[sid][0][elem] = conv_activation(val);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                auto out_tile_blk = MatLayoutLSTM::tnc_block(out_buf, chunk_size_out, num_chunks,
                                                             out_size, ts, m_blk, n_blk);
                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    for (int j = 0; j < SIMD_TILES_N; ++j) {
                        simdgroup_ftype8x8 A;
                        simdgroup_load(A, simd_out_buf[sid][i * SIMD_TILES_N + j], TILE_SIZE);
                        out_tile_blk.store(A, i, j);
                    }
                }
            }
        }
    }
}
#undef SIMD_GROUPS

struct LstmArgs {
    int batch_tiles;
    int chunk_size;
    int time_step_begin;  // Inclusive
    int time_step_end;    // Exclusive
};

template <typename InLayout, typename OutLayout>
kernel void reorder(const device LstmArgs* const args,
                    const device typename InLayout::ftype* const in,
                    device typename OutLayout::ftype* const out,
                    KERNEL_INDEX_INPUTS) {
    threadgroup typename InLayout::ftype bfr_in[32][TILE_SIZE * TILE_SIZE];
    threadgroup typename OutLayout::ftype bfr_out[32][TILE_SIZE * TILE_SIZE];
    const int batch_size = args->batch_tiles * TILE_SIZE;
    const int chunk_size = args->chunk_size;
    const int m_blks = args->batch_tiles / SIMD_TILES_M;
    const int n_blks = kLstmLayerSize / (SIMD_TILES_N * TILE_SIZE);
    uint lane_id = tid & 31;

    OutLayout::zero_initial_state(out, chunk_size, batch_size, kLstmLayerSize, gid, threadgroups,
                                  sid, simdgroups);
    for (int ts = 0; ts < chunk_size; ++ts) {
        for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
            for (int n_blk = sid; n_blk < n_blks; n_blk += simdgroups) {
                auto in_blk = InLayout::tnc_block(in, chunk_size, batch_size, kLstmLayerSize, ts,
                                                  m_blk, n_blk);
                auto out_blk = OutLayout::tnc_block(out, chunk_size, batch_size, kLstmLayerSize, ts,
                                                    m_blk, n_blk);
                for (int m_tile = 0; m_tile < SIMD_TILES_M; ++m_tile) {
                    for (int n_tile = 0; n_tile < SIMD_TILES_N; ++n_tile) {
                        typename decltype(in_blk)::simdgroup_tile in_tile;
                        typename decltype(out_blk)::simdgroup_tile out_tile;
                        in_blk.load(in_tile, m_tile, n_tile);
                        simdgroup_store(in_tile, bfr_in[sid], TILE_SIZE);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        for (int i = lane_id; i < TILE_SIZE * TILE_SIZE; i += 32) {
                            bfr_out[sid][i] = ftype(bfr_in[sid][i]);
                        }
                        simdgroup_load(out_tile, bfr_out[sid], TILE_SIZE);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        out_blk.store(out_tile, m_tile, n_tile);
                    }
                }
            }
        }
    }
}

template [[host_name("reorder_input_to_fwd_lstm_output")]] kernel void
reorder<MatLayoutRowMajor<SIMD_TILES_M, SIMD_TILES_N, ftype_in>,
        MatLayoutLSTM<SIMD_TILES_M, SIMD_TILES_N, FORWARD_LSTM_OUTPUT, ftype>>(
        const device LstmArgs*,
        const device ftype_in*,
        device ftype*,
        KERNEL_INDEX_INPUTS);

template [[host_name("reorder_input_to_rev_lstm_output")]] kernel void
reorder<MatLayoutRowMajor<SIMD_TILES_M, SIMD_TILES_N, ftype_in>,
        MatLayoutLSTM<SIMD_TILES_M, SIMD_TILES_N, REVERSE_LSTM_OUTPUT, ftype>>(
        const device LstmArgs*,
        const device ftype_in*,
        device ftype*,
        KERNEL_INDEX_INPUTS);

template [[host_name("reorder_rev_lstm_output_to_linear")]] kernel void
reorder<MatLayoutLSTM<SIMD_TILES_M, SIMD_TILES_N, REVERSE_LSTM_OUTPUT, ftype>,
        MatLayoutRowMajor<SIMD_TILES_M, SIMD_TILES_N, ftype_out>>(const device LstmArgs*,
                                                                  const device ftype*,
                                                                  device ftype_out*,
                                                                  KERNEL_INDEX_INPUTS);

// Note: max_total_threads_per_threadgroup is set via ComputePipelineDescriptor,
// rather than an attribute here, since it depends on the SIMD group count,
// which varies according to LSTM layer size.
kernel void lstm(const device LstmArgs* const args,
                 device ftype* const in_out,
                 const device ftype* const weights_buf,
                 device ftype* const state_buf,
                 // The sizes of these buffers are set via MTL::ComputeCommandEncoder.
                 // They depend on the SIMD group count.
                 threadgroup ftype (*const simd_res_buf)[2 * TILE_SIZE * TILE_SIZE],
                 threadgroup ftype (*const simd_out_buf)[TILE_SIZE * TILE_SIZE],
                 KERNEL_INDEX_INPUTS) {
    const int chunk_size = args->chunk_size;
    const int batch_tiles = args->batch_tiles;
    const int time_step_begin = args->time_step_begin;
    const int time_step_end = args->time_step_end;
    const int m_blks = batch_tiles / SIMD_TILES_M;
    const int n_blks = kLstmLayerSize * 4 / (TILE_SIZE * SIMD_TILES_N);
    const int k_tiles = kLstmLayerSize / TILE_SIZE;
    const int batch_size = batch_tiles * TILE_SIZE;
    const int w_stride = kLstmLayerSize * 4;
    MatMul<SIMD_TILES_M, SIMD_TILES_N> mm;
    using MatLayoutLSTM = MatLayoutLSTM<SIMD_TILES_M, SIMD_TILES_N, NO_OFFSET>;
    const device ftype* const bias = weights_buf + 3 * kLstmLayerSize * w_stride;

    const uint t_idx = tid & 31;
    const uint col_bits = t_idx & 3;
    const uint row = t_idx >> 2;
    const uint rb_idx = t_idx * 4;

    if (time_step_begin == 0) {
        for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
            for (int chunk = tid; chunk < SIMD_TILES_M * TILE_SIZE; chunk += threads) {
                for (int i = 0; i < kLstmLayerSize; ++i) {
                    state_buf[i * batch_tiles * TILE_SIZE + m_blk * SIMD_TILES_M * TILE_SIZE +
                              chunk] = 0;
                }
            }
        }
    }

    for (int iter = time_step_begin; iter < time_step_end; ++iter) {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const int timestep_in = kLstmReversedInTime ? chunk_size - iter : iter;
        for (int m_blk = gid; m_blk < m_blks; m_blk += threadgroups) {
            for (int n_blk = sid; n_blk < n_blks; n_blk += simdgroups) {
                mm.load_bias(bias, n_blk * SIMD_TILES_N * TILE_SIZE);
                auto mat_a = MatLayoutLSTM::tnc_block(in_out, 0, batch_size, kLstmLayerSize,
                                                      timestep_in, m_blk, 0);
                TileBlock<RowMajor> mat_b(weights_buf, w_stride, 0,
                                          n_blk * SIMD_TILES_N * TILE_SIZE);
                auto mat_c = MatLayoutLSTM::tnc_block(in_out, 0, batch_size, kLstmLayerSize,
                                                      timestep_in + 1, m_blk, 0);
                // Surprisingly, executing the second mma starting from 2*k_tiles is faster than
                // doing `mat_a = MatLayoutLSTM::tnc_block(..., timestep_in+2, ...);` or adding
                // an offset to `mat_a.ptr` and `mat_b.ptr`
                // This is why `weights_buf` has `3*kLstmLayerSize` rows instead of `2*kLstmLayerSize` rows
                mm.mma(0, k_tiles, mat_a, mat_b);
                mm.mma(2 * k_tiles, 3 * k_tiles, mat_a, mat_b);

                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    const uint chunk_idx = (m_blk * SIMD_TILES_M + i) * TILE_SIZE + row;
                    for (int j = 0; j < SIMD_TILES_N; j += 2) {
                        simdgroup_store(mm.acc(i, j + 0), simd_res_buf[sid], 2 * TILE_SIZE);
                        simdgroup_store(mm.acc(i, j + 1), simd_res_buf[sid] + TILE_SIZE,
                                        2 * TILE_SIZE);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        const uint col = j * 2 + col_bits;
                        const uint out_col = n_blk * SIMD_TILES_N * 2 + col;
                        const uint out_idx = out_col * batch_size + chunk_idx;
                        const float g = tanh_fast(simd_res_buf[sid][rb_idx + 0]);
                        const float i = sigmoid(simd_res_buf[sid][rb_idx + 1]);
                        const float f = sigmoid(simd_res_buf[sid][rb_idx + 2]);
                        const float o = sigmoid(simd_res_buf[sid][rb_idx + 3]);
                        const float state = f * state_buf[out_idx] + i * g;
                        const float h = o * tanh_fast(state);
                        state_buf[out_idx] = state;
                        simd_out_buf[sid][row * SIMD_TILES_N * 2 + col] = h;
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_ftype8x8 A;
                    for (int j = 0; j < SIMD_TILES_N / 4; ++j) {
                        simdgroup_load(A, simd_out_buf[sid], SIMD_TILES_N * 2,
                                       ulong2(j * TILE_SIZE, 0));
                        mat_c.store(A, i, n_blk * (SIMD_TILES_N / 4) + j);
                    }
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

template <typename InputMatLayout>
kernel void linear(const device LinearArgs* const args,
                   const device ftype* const in_buf,
                   const device ftype* const weights_buf,
                   device void* const out_buf,
                   // The size of this buffer is set via MTL::ComputeCommandEncoder.
                   // It depends on the SIMD group count.
                   threadgroup ftype (*const simd_out_buf)[TILE_SIZE * TILE_SIZE],
                   KERNEL_INDEX_INPUTS) {
    const int chunk_size = args->chunk_size;
    const int in_batch_size = args->in_batch_tiles * TILE_SIZE;
    const int in_batch_block_offset = args->in_batch_tile_offset / SIMD_TILES_M;
    const int out_batch_tiles = args->out_batch_tiles;
    const int m_blks = out_batch_tiles / SIMD_TILES_M;
    const int n_blks = kLinearOutSize / (TILE_SIZE * SIMD_TILES_N);
    const int k_tiles = kLinearInSize / TILE_SIZE;
    const int w_stride = kLinearOutSize;
    const int out_stride = kLinearOutSize;
    const device ftype* const bias = weights_buf + kLinearInSize * w_stride;
    MatMul<SIMD_TILES_M, SIMD_TILES_N> mm;

    for (int ts = gid; ts < chunk_size; ts += threadgroups) {
        auto out_buf_offset = ts * kLinearOutSize * out_batch_tiles * TILE_SIZE;
        device auto* const out_int8 = (device int8_t*)out_buf + out_buf_offset;
        device auto* const out_ftype = (device ftype*)out_buf + out_buf_offset;

        for (int m_blk = 0; m_blk < m_blks; ++m_blk) {
            for (int n_blk = sid; n_blk < n_blks; n_blk += simdgroups) {
                mm.load_bias(bias, n_blk * SIMD_TILES_N * TILE_SIZE);
                auto mat_a =
                        InputMatLayout::tnc_block(in_buf, chunk_size, in_batch_size, kLinearInSize,
                                                  ts, m_blk + in_batch_block_offset, 0);
                TileBlock<RowMajor> mat_b(weights_buf, w_stride, 0,
                                          n_blk * SIMD_TILES_N * TILE_SIZE);
                mm.mma(0, k_tiles, mat_a, mat_b);
                for (int i = 0; i < SIMD_TILES_M; ++i) {
                    for (int j = 0; j < SIMD_TILES_N; ++j) {
                        // Store this 8x8 tile to threadgroup memory as ftype.
                        simdgroup_store(mm.acc(i, j), simd_out_buf[sid], TILE_SIZE);

                        const uint tile_i = (m_blk * SIMD_TILES_M + i) * TILE_SIZE;
                        const uint tile_j = (n_blk * SIMD_TILES_N + j) * TILE_SIZE;

                        // Apply tanh activation or clamping, scaling, and type conversion.
                        // Store to the output buffer.
                        for (int elem = tid & 31; elem < TILE_SIZE * TILE_SIZE; elem += 32) {
                            const ftype matmul_output = simd_out_buf[sid][elem];
                            const auto with_clamp =
                                    kLinearOutputClamp
                                            ? clamp(matmul_output, ftype(-5.0f), ftype(5.0f))
                                            : matmul_output;
                            const auto with_tanh =
                                    kLinearOutputTanh ? tanh_fast(with_clamp) : with_clamp;
                            const auto with_scale = with_tanh * kLinearOutputScale;

                            const int in_tile_i = elem / TILE_SIZE;
                            const int in_tile_j = elem % TILE_SIZE;
                            if (kLinearOutputAsByte) {
                                out_int8[(tile_i + in_tile_i) * out_stride + tile_j + in_tile_j] =
                                        static_cast<int8_t>(with_scale);
                            } else {
                                out_ftype[(tile_i + in_tile_i) * out_stride + tile_j + in_tile_j] =
                                        static_cast<ftype>(with_scale);
                            }
                        }
                    }
                }
            }
        }
    }
}

template [[host_name("linear")]] kernel void linear<MatLayoutRowMajor<SIMD_TILES_M, SIMD_TILES_N>>(
        const device LinearArgs*,
        const device ftype*,
        const device ftype*,
        device void* const,
        threadgroup ftype (*const simd_out_buf)[TILE_SIZE * TILE_SIZE],
        KERNEL_INDEX_INPUTS);

template [[host_name("linear_from_rev_lstm")]] kernel void
linear<MatLayoutLSTM<SIMD_TILES_M, SIMD_TILES_N, REVERSE_LSTM_OUTPUT>>(
        const device LinearArgs*,
        const device ftype*,
        const device ftype*,
        device void* const,
        threadgroup ftype (*const simd_out_buf)[TILE_SIZE * TILE_SIZE],
        KERNEL_INDEX_INPUTS);
