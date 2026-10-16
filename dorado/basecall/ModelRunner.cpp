#include "basecall/ModelRunner.h"

#include "basecall/ModelCaller.h"
#include "config/BasecallModelConfig.h"
#include "utils/log_utils.h"
#include "utils/memory_utils.h"
#include "utils/string_utils.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace dorado::basecall {

ModelRunner::ModelRunner(std::shared_ptr<ModelCaller> caller, size_t batch_dims_idx)
        : m_caller(std::move(caller)),
          m_runner_context(m_caller->create_runner_context(batch_dims_idx)) {}

ModelRunner::~ModelRunner() = default;

std::vector<decode::DecodedChunk> ModelRunner::call_chunks(int num_chunks) {
    ++m_num_batches_called;
    return m_runner_context->call_chunks(num_chunks);
}

void ModelRunner::accept_chunk(int chunk_idx, const at::Tensor & chunk_CT) {
    m_runner_context->accept_chunk(chunk_idx, chunk_CT);
}

const config::BasecallModelConfig & ModelRunner::config() const { return m_caller->config(); }

std::shared_ptr<ModBaseInfo> ModelRunner::modbase_info() const { return m_caller->modbase_info(); };

size_t ModelRunner::chunk_size() const { return m_runner_context->chunk_size(); }

size_t ModelRunner::batch_size() const { return m_runner_context->batch_size(); }

bool ModelRunner::variable_chunk_sizes() const { return m_runner_context->variable_chunk_sizes(); }

std::pair<int, int> ModelRunner::batch_timeouts_ms() const { return m_caller->batch_timeouts_ms(); }

bool ModelRunner::is_low_latency() const { return m_caller->is_low_latency(); }

void ModelRunner::terminate() { m_caller->terminate(); }

void ModelRunner::restart() { m_caller->restart(); }

std::string ModelRunner::get_name() const {
    std::ostringstream name;
    name << "ModelRunner_" << this;
    return name.str();
}

stats::NamedStats ModelRunner::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(*m_caller);
    stats["batches_called"] = double(m_num_batches_called);
    return stats;
}

size_t auto_calculate_num_runners(const config::BasecallModelConfig & model_config,
                                  float memory_fraction) {
    const auto model_name = model_config.model_name();

    if (const char * num_runners_str = getenv("DORADO_CPU_RUNNERS"); num_runners_str != nullptr) {
        auto num_runners = utils::from_chars<size_t>(num_runners_str);
        if (num_runners) {
            DORADO_LOG(info, "Overriding CPU runners to {}", num_runners.value());
            return num_runners.value();
        }
        throw std::runtime_error(fmt::format("Invalid DORADO_CPU_RUNNERS: '{}'", num_runners_str));
    }

    // These values were determined empirically by running 1, 2, 4 and 8 runners per model.
    std::optional<double> required_ram_per_runner_GB;
    if (model_config.is_lstm_model()) {
        if (model_name.find("_fast@v") != std::string::npos) {
            required_ram_per_runner_GB = 1.5;
        } else if (model_name.find("_hac@v") != std::string::npos) {
            required_ram_per_runner_GB = 4.5;
        } else if (model_name.find("_sup@v") != std::string::npos) {
            required_ram_per_runner_GB = 12.5;
        }
    } else if (model_config.is_flstm_model() && model_name.find("_hac@v") != std::string::npos) {
        required_ram_per_runner_GB = 10;
    }
    if (!required_ram_per_runner_GB.has_value()) {
        return 1;
    }

    assert(model_config.basecaller.batch_size() > 0);
    required_ram_per_runner_GB.value() *= model_config.basecaller.batch_size() / 128.f;

    const auto free_ram_GB =
            static_cast<size_t>(utils::available_host_memory_GB()) * memory_fraction;
    const auto num_runners = static_cast<size_t>(free_ram_GB / required_ram_per_runner_GB.value());
    return std::clamp(num_runners, size_t(1), std::size_t(std::thread::hardware_concurrency()));
}

}  // namespace dorado::basecall
