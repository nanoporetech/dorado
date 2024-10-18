#include "FakeDataLoader.h"

#include "ReadPipeline.h"

#include <ATen/Functions.h>

#include <cstdint>
#include <memory>

namespace dorado {

void FakeDataLoader::load_reads(const int num_reads) {
    for (int i = 0; i < num_reads; ++i) {
        auto fake_read = std::make_unique<SimplexRead>();

        constexpr int64_t read_size = 40000;
        fake_read->read_common.raw_data = at::randint(0, 10000, {read_size}, at::kShort);
        fake_read->read_common.read_id = "Placeholder-read-id";

        m_pipeline.push_message(std::move(fake_read));
    }
}

FakeDataLoader::FakeDataLoader(Pipeline& pipeline) : m_pipeline(pipeline) {}

}  // namespace dorado
