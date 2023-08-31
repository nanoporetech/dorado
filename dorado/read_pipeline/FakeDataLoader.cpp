#include "FakeDataLoader.h"

#include "read_pipeline/ReadPipeline.h"

#include <torch/torch.h>

#include <cstdint>
#include <memory>

namespace dorado {

void FakeDataLoader::load_reads(const int num_reads) {
    for (int i = 0; i < num_reads; ++i) {
        std::shared_ptr<Read> fake_read = std::make_shared<Read>();

        constexpr int64_t read_size = 40000;
        fake_read->raw_data = torch::randint(0, 10000, {read_size}, torch::kInt16);
        fake_read->read_id = "Placeholder-read-id";

        m_pipeline.push_message(std::move(fake_read));
    }
}

FakeDataLoader::FakeDataLoader(Pipeline& pipeline) : m_pipeline(pipeline) {}

}  // namespace dorado
