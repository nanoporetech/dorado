#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

class ReadToBamType : public MessageSink {
public:
    ReadToBamType(bool emit_moves,
                  bool rna,
                  size_t num_worker_threads,
                  float modbase_threshold_frac = 0,
                  size_t max_reads = 1000);
    ~ReadToBamType() { terminate_impl(); }
    std::string get_name() const override { return "ReadToBamType"; }
    void terminate() override { terminate_impl(); };

private:
    void terminate_impl();
    void worker_thread();

    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<size_t> m_active_threads;

    bool m_emit_moves;
    bool m_rna;
    uint8_t m_modbase_threshold;
};

}  // namespace dorado
