#pragma once

#include <torch/torch.h>

#include <memory>
#include <vector>

namespace dorado {

class Read;

class RemoraUtils {
public:
    static constexpr int NUM_BASES = 4;
    static const std::vector<int> BASE_IDS;
};

struct RemoraChunk {
    RemoraChunk(std::shared_ptr<Read> read,
                torch::Tensor input_signal,
                std::vector<float> kmer_data,
                size_t position)
            : source_read(read),
              signal(input_signal),
              encoded_kmers(std::move(kmer_data)),
              context_hit(position) {}

    std::weak_ptr<Read> source_read;
    torch::Tensor signal;
    std::vector<float> encoded_kmers;
    size_t context_hit;
    std::vector<float> scores;
};

}  // namespace dorado