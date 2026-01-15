#include "read_pipeline/nodes/PolyACalculatorNode.h"

#include "poly_tail/poly_tail_calculator.h"
#include "poly_tail/poly_tail_calculator_selector.h"
#include "read_pipeline/base/ClientInfo.h"
#include "utils/context_container.h"

#include <spdlog/spdlog.h>

namespace {
constexpr std::size_t MAX_INPUT_QUEUE_SIZE{10000};
constexpr std::size_t MAX_PROCESSING_QUEUE_SIZE{MAX_INPUT_QUEUE_SIZE / 2};
}  // namespace

namespace dorado {

void PolyACalculatorNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<SimplexReadPtr>(std::move(message));
        m_task_executor.send([this, read_ = std::move(read)]() mutable {
            process_read(*read_);
            send_message_to_sink(std::move(read_));
        });
    }
}

void PolyACalculatorNode::process_read(SimplexRead &read) {
    auto selector = read.read_common.client_info->contexts()
                            .get_ptr<const poly_tail::PolyTailCalculatorSelector>();

    if (!selector) {
        num_not_called++;
        return;
    }

    auto calculator = selector->get_calculator(read.read_common.barcode);
    if (!calculator || !calculator->enabled()) {
        // Barcode overrides have been specified and this read is either unclassified
        // or estimation for this barcode has been explicitly disabled
        num_not_called++;
        return;
    }

    // Poly-tail selection is enabled, so adjust default value of poly-tail length and signal ranges
    read.read_common.rna_poly_tail_length = ReadCommon::POLY_TAIL_NOT_FOUND;
    read.read_common.poly_tail_signal_anchor = ReadCommon::POLY_TAIL_NOT_FOUND;
    read.read_common.poly_tail_signal_boundaries = std::array{
            std::make_pair(ReadCommon::POLY_TAIL_NOT_FOUND, ReadCommon::POLY_TAIL_NOT_FOUND),
            std::make_pair(ReadCommon::POLY_TAIL_NOT_FOUND, ReadCommon::POLY_TAIL_NOT_FOUND)};

    auto signal_info = calculator->determine_signal_anchor_and_strand(read);
    if (!std::empty(signal_info)) {
        read.read_common.poly_tail_signal_anchor =
                signal_info[0].signal_anchor + read.read_common.num_trimmed_samples;

        auto polya_tail_info = calculator->calculate_num_bases(read, signal_info);

        read.read_common.poly_tail_signal_boundaries[0] = polya_tail_info.signal_range;
        read.read_common.poly_tail_signal_boundaries[1] = polya_tail_info.split_signal_range;

        if (polya_tail_info.num_bases > 0 &&
            polya_tail_info.num_bases < calculator->max_tail_length()) {
            // Update debug stats.
            total_tail_lengths_called += polya_tail_info.num_bases;
            ++num_called;
            if (spdlog::get_level() <= spdlog::level::debug) {
                std::lock_guard<std::mutex> lock(m_mutex);
                tail_length_counts[polya_tail_info.num_bases]++;
            }
            // Set tail length property in the read.
            read.read_common.rna_poly_tail_length = polya_tail_info.num_bases;
        } else {
            // On failure, set tail length to 0 to distinguish from the anchor not having been found.
            read.read_common.rna_poly_tail_length = 0;
            num_not_called++;
        }
    } else {
        num_not_called++;
    }
}

PolyACalculatorNode::PolyACalculatorNode(
        std::shared_ptr<utils::concurrency::MultiQueueThreadPool> thread_pool,
        utils::concurrency::TaskPriority pipeline_priority,
        size_t max_reads)
        : MessageSink(max_reads, 1),
          m_thread_pool(std::move(thread_pool)),
          m_task_executor(*m_thread_pool, pipeline_priority, MAX_PROCESSING_QUEUE_SIZE) {}

PolyACalculatorNode::PolyACalculatorNode(size_t threads, size_t max_reads)
        : PolyACalculatorNode(
                  std::make_shared<utils::concurrency::MultiQueueThreadPool>(threads, "polya_pool"),
                  utils::concurrency::TaskPriority::normal,
                  max_reads) {}

PolyACalculatorNode::~PolyACalculatorNode() { terminate_impl(utils::AsyncQueueTerminateFast::Yes); }

std::string PolyACalculatorNode::get_name() const { return "PolyACalculator"; }

void PolyACalculatorNode::terminate_impl(utils::AsyncQueueTerminateFast fast) {
    stop_input_processing(fast);
}

void PolyACalculatorNode::terminate(const TerminateOptions &terminate_options) {
    terminate_impl(terminate_options.fast);
    m_task_executor.flush();
};

void PolyACalculatorNode::restart() {
    m_task_executor.restart();
    start_input_processing([this] { input_thread_fn(); }, "polyacalc_node");
}

stats::NamedStats PolyACalculatorNode::sample_stats() const {
    stats::NamedStats stats = MessageSink::sample_stats();
    stats["queued_tasks"] = double(m_task_executor.num_tasks_in_flight());
    stats["reads_not_estimated"] = static_cast<double>(num_not_called.load());
    stats["reads_estimated"] = static_cast<double>(num_called.load());
    stats["average_tail_length"] = static_cast<double>(
            num_called.load() > 0 ? total_tail_lengths_called.load() / num_called.load() : 0);

    if (spdlog::get_level() <= spdlog::level::debug) {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto &[len, count] : tail_length_counts) {
            std::string key = "pt." + std::to_string(len);
            stats[key] = static_cast<float>(count);
        }
    }

    return stats;
}

}  // namespace dorado
