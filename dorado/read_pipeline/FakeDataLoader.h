#pragma once

namespace dorado {

class MessageSink;

// Supplies a stream of reads with random signals for testing purposes.
class FakeDataLoader {
public:
    FakeDataLoader(MessageSink& read_sink);
    void load_reads(int num_reads);

private:
    MessageSink& m_read_sink;
};

}  // namespace dorado
