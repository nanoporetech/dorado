#pragma once

namespace dorado::nn::metal {

struct ScanArgs {
    int T;
    int N;
    int C;
};
static_assert(sizeof(ScanArgs) == 12);

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
static_assert(sizeof(ConvArgs) == 36);

struct LstmArgs {
    int batch_tiles;
    int chunk_size;
    int time_step_begin;  // Inclusive
    int time_step_end;    // Exclusive
};
static_assert(sizeof(LstmArgs) == 16);

struct LinearArgs {
    int in_batch_tiles;
    int in_batch_tile_offset;
    int out_batch_tiles;
    int chunk_size;
};
static_assert(sizeof(LinearArgs) == 16);

}  // namespace dorado::nn::metal
