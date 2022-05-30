#pragma once

#include <torch/torch.h>

struct DecodedChunk {
    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;
};

struct DecoderOptions {
    size_t beam_width = 32;
    float beam_cut = 100.0;
    float blank_score = 2.0;
    float q_shift = 0.0;
    float q_scale = 1.0;
    float temperature = 1.0;
    bool move_pad = false;
};

class Decoder {
public:
    virtual std::vector<DecodedChunk> beam_search(torch::Tensor scores,
                                                  int num_chunks,
                                                  DecoderOptions options) = 0;
};
