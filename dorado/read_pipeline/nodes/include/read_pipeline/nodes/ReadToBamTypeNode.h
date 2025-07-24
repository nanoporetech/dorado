#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace dorado {

namespace utils {
class SampleSheet;
}

class ReadToBamTypeNode : public MessageSink {
public:
    ReadToBamTypeNode(bool emit_moves,
                      size_t num_worker_threads,
                      std::optional<float> modbase_threshold_frac,
                      std::shared_ptr<const utils::SampleSheet> sample_sheet,
                      size_t max_reads,
                      size_t min_qscore);
    ~ReadToBamTypeNode();

    std::string get_name() const override;
    void terminate(const TerminateOptions &) override;
    void restart() override;

    // TODO: refactor duplex.cpp pipeline setup so that this isn't required.
    void set_modbase_threshold(float threshold);

private:
    void input_thread_fn();

    bool m_emit_moves;
    std::optional<uint8_t> m_modbase_threshold;
    std::shared_ptr<const utils::SampleSheet> m_sample_sheet;
    size_t m_min_qscore;
};

}  // namespace dorado
