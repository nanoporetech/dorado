#pragma once

namespace dorado {

class Pipeline;

// Supplies a stream of reads with random signals for testing purposes.
class FakeDataLoader {
public:
    FakeDataLoader(Pipeline& read_sink);
    void load_reads(int num_reads);

private:
    Pipeline& m_pipeline;
};

}  // namespace dorado
