#pragma once

#include "utils/parameters.h"

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>

namespace dorado::config {

// Stores basecaller parameters and manages the overwrite priority from various setters
// where Default < Config < CLI Argument < Forced.
class BasecallerParams {
public:
    // Overwrite priority
    enum class Priority : int {
        DEFAULT = 0,
        CONFIG = 1,
        CLI_ARG = 2,
        FORCE = 3,
    };

    BasecallerParams() {}
    BasecallerParams(const std::filesystem::path& path) { update(path); };
    ~BasecallerParams() {}

    // Number of signal samples in a chunk
    int chunk_size() const { return m_chunk_size.val; }
    // Number of signal samples overlapping consecutive chunks
    int overlap() const { return m_overlap.val; }
    // Number of chunks in a batch
    int batch_size() const { return m_batch_size.val; }

    // Set chunk_size with the FORCE priority.
    void set_chunk_size(int chunk_size) {
        set_value(m_chunk_size, Value{chunk_size, Priority::FORCE});
    }
    // Set overlap with the FORCE priority.
    void set_overlap(int overlap) { set_value(m_overlap, Value{overlap, Priority::FORCE}); }
    // Set batch_size with the FORCE priority.
    void set_batch_size(int batch_size) {
        set_value(m_batch_size, Value{batch_size, Priority::FORCE});
    }

    // Update parameters if present in the model config.toml in path dir
    void update(const std::filesystem::path& path);

    // Update set values if priority is greater than or equal to exising priority
    void update(Priority priority,
                std::optional<int> chunk_size,
                std::optional<int> overlap,
                std::optional<int> batch_size);

    // Update values from other basecaller params, self takes priory if tied unless forced
    void update(const BasecallerParams& other);

    // Normalise `chunk_size` and `overlap` so that `overlap` is a multiple of `stride`, and
    // `chunk_size` is both greater than `overlap` and a multiple of `chunk_size_granularity`.
    void normalise(int chunk_size_granularity, int stride);

    std::string to_string() const {
        std::string str = "BasecallerParams {";
        str += " chunk_size:" + std::to_string(m_chunk_size.val);
        str += " overlap:" + std::to_string(m_overlap.val);
        str += " batch_size:" + std::to_string(m_batch_size.val);
        str += "}";
        return str;
    }

protected:
    struct Value {
        int val;
        Priority priority{Priority::DEFAULT};
    };

    Value m_chunk_size{utils::default_parameters.chunksize, Priority::DEFAULT};
    Value m_overlap{utils::default_parameters.overlap, Priority::DEFAULT};
    Value m_batch_size{utils::default_parameters.batchsize, Priority::DEFAULT};

    // Set self to other if other has higher or FORCE priority.
    // Returns false - no change, true - updated
    bool set_value(Value& self, const Value& other);
};

}  // namespace dorado::config