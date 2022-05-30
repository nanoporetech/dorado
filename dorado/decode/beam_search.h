#pragma once

#include "torch/torch.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

template <typename T>
void merge_sort(T* data,
                const size_t count,
                const size_t cutoff,
                bool (*less_func)(const T&, const T&)) {
    std::vector<T> working_buff(count);
    T* source_buff = data;
    T* dest_buff = working_buff.data();
    for (size_t src_block_size = 1; src_block_size < count; src_block_size *= 2) {
        // Merge source blocks
        for (size_t start_idx = 0; start_idx < count; start_idx += src_block_size * 2) {
            // Merge_blocks (in parallel)

            T* first_blk = source_buff + start_idx;
            T* second_blk_start = first_blk + src_block_size;
            T* second_blk = second_blk_start;
            size_t end_merge_idx = std::min(
                    count,
                    start_idx + src_block_size *
                                        2);  // idx of to element after the last one we should merge
            // Don't write more than 'cutoff' values into dest (useful if we only need 'cutoff' best results)
            size_t end_write_idx = std::min(end_merge_idx, start_idx + cutoff);

            for (T* dest = dest_buff + start_idx; dest < dest_buff + end_write_idx; dest++) {
                if (first_blk == second_blk_start  // All of first block consumed
                    || (second_blk < source_buff + end_merge_idx &&
                        less_func(*second_blk,
                                  *first_blk))) {  // second block available and should go first
                    // Write from second blk
                    *dest = *second_blk;
                    second_blk++;
                } else {
                    // Write from first blk
                    *dest = *first_blk;
                    first_blk++;
                }
            }
        }
        std::swap(source_buff, dest_buff);
    }

    // If we do an odd number of loops on block size, the destination buffer will not be data, so we need to copy back
    // NOTE: the last loop's destination buff is now source_buff
    if (source_buff != data) {
        for (size_t i = 0; i < count; i++) {
            data[i] = source_buff[i];
        }
    }
}

std::tuple<std::string, std::string, std::vector<uint8_t>> beam_search_decode(
        const torch::Tensor scores_t,
        const torch::Tensor back_guides_t,
        const torch::Tensor posts_t,
        size_t beam_width = 5,
        float beam_cut = 100.0,
        float fixed_stay_score = 2.0,
        float q_shift = 0.0,
        float q_scale = 1.0,
        float temperature = 1.0);
