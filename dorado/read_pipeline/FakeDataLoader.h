#pragma once

namespace dorado {

class ReadSink;

// Supplies a stream of reads with random signals for testing purposes.
class FakeDataLoader {
public:
    FakeDataLoader(ReadSink& read_sink);
    void load_reads(int num_reads);

private:
    ReadSink& m_read_sink;
};

}  // namespace dorado
