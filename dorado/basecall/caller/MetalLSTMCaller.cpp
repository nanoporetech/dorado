#include "MetalLSTMCaller.h"

#include "MetalCallerTask.h"
#include "basecall/crf_utils.h"
#include "decode/beam_search.h"
#include "model/MetalCRFModel.h"
#include "torch_utils/metal_utils.h"
#include "utils/math_utils.h"
#include "utils/memory_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>

using namespace dorado::utils;
using torch::indexing::Slice;

namespace {
constexpr int MTL_CORE_BATCH_SIZE = 48;

constexpr at::ScalarType scores_dtype = at::kChar;
constexpr at::ScalarType posts_dtype = at::kShort;

CREATE_POINT_OF_INTEREST_ID(MetalLSTMCaller);
}  // namespace

namespace dorado::basecall {

using namespace config;

MetalLSTMCaller::MetalLSTMCaller(const BasecallModelConfig &model_config,
                                 float memory_limit_fraction)
        : MetalCaller(model_config) {
    ScopedAutoReleasePool autorelease_pool;
    // Our metal builds assume shared memory, so it's safe to check host.
    if (auto total_mem = utils::total_host_memory_GB(); total_mem < 16) {
        DORADO_LOG(warn,
                   "Less than 16GB of memory available: {}GB detected. "
                   "This is below minimum spec and may cause issues",
                   total_mem);
    }

    m_device = get_mtl_device();

    m_decoder_options = decode::DecoderOptions();
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;

    // TODO -- we don't honour the config n_base
    constexpr int n_base = 4;
    m_states = pow(n_base, model_config.state_len);

    auto state_dict = load_crf_model_weights(model_config);

    assert(model_config.has_normalised_basecaller_params());
    const auto chunk_size = model_config.basecaller.chunk_size();
    const auto batch_size = model_config.basecaller.batch_size();
    auto selected_batch_size = (batch_size == 0) ? benchmark_batch_sizes(model_config, state_dict,
                                                                         memory_limit_fraction)
                                                 : utils::pad_to(batch_size, MTL_CORE_BATCH_SIZE);

    set_chunk_batch_size(model_config, state_dict, chunk_size, selected_batch_size);
    start_threads();
}

MetalLSTMCaller::~MetalLSTMCaller() = default;

at::Tensor MetalLSTMCaller::create_input_tensor() const {
    // Metal convolution kernels operate with channel ordering (N, T, C).  If m_input
    // is to be submitted directly then it must also have this arrangement.
    // Note that this is not the same as other caller implementations, which
    // have T innermost.
    return at::zeros({m_batch_size, m_in_chunk_size, m_config.num_features}, at::kHalf);
}

int MetalLSTMCaller::get_max_safe_batch_size(float memory_limit_fraction,
                                             const config::BasecallModelConfig &model_config) {
    const size_t physical_memory = get_apple_physical_memory_bytes();
    const size_t usable_memory = physical_memory * memory_limit_fraction;
    DORADO_LOG(debug, "Physical/Usable memory available: {}/{} GB", physical_memory / BYTES_PER_GB,
               usable_memory / BYTES_PER_GB);

    // Constrain the maximum batch size to use about half physical memory for decode buffers,
    // with neural network GPU buffers and CPU buffers assumed to occupy a subset of the
    // remaining memory.  This generally constrains the batch size to use fewer than
    // the maximum GPU cores when running sup models on systems with a large GPU core
    // to system memory ratio.
    const auto chunk_size = model_config.basecaller.chunk_size();
    const auto out_chunk_size = chunk_size / model_config.stride;

    // TODO -- we don't honour the config n_base
    constexpr int n_base = 4;
    const int states = pow(n_base, model_config.state_len);

    const auto decode_buffer_size_per_elem =
            static_cast<size_t>(out_chunk_size) *
            (static_cast<size_t>(model_config.outsize) +      // Scores
             static_cast<size_t>(states) * sizeof(int16_t) +  // Posts
             static_cast<size_t>(states) * sizeof(float));    // Back guides.
    DORADO_LOG(trace, "decode_buffer_size_per_elem {}", decode_buffer_size_per_elem);
    const int max_batch_size = static_cast<int>(
            std::clamp(utils::pad_to(usable_memory / (2 * decode_buffer_size_per_elem),
                                     static_cast<size_t>(MTL_CORE_BATCH_SIZE)),
                       static_cast<size_t>(MTL_CORE_BATCH_SIZE),
                       static_cast<size_t>(MTL_CORE_BATCH_SIZE * get_mtl_device_core_count())));
    DORADO_LOG(trace, "max_batch_size {}", max_batch_size);
    return max_batch_size;
}

int MetalLSTMCaller::get_batch_size_granularity() { return MTL_CORE_BATCH_SIZE; }

void MetalLSTMCaller::set_chunk_batch_size(const BasecallModelConfig &model_config,
                                           const std::vector<at::Tensor> &state_dict,
                                           int chunk_size,
                                           int batch_size) {
    // Chunk size already normalised to inner stride
    m_in_chunk_size = chunk_size;
    m_out_chunk_size = chunk_size / model_config.stride;

    m_batch_size = batch_size;

    // Allocations beyond 4GB can fail, and the linear layer output buffer
    // hits this limit with batch sizes larger than 384 with typical
    // chunk sizes.  We also want to limit memory usage in general.
    // At the same time, the LSTM layer performance benefits
    // from large batch sizes.
    // We therefore run the linear layer via 1 or more kernel runs, each
    // with an output buffer of limited size, aiming for <= kMaxBufferSize.
    // As things stand, we need an exactly even split of batch elements in
    // the linear layer output buffers (this could be relaxed).
    // We therefore want the smallest divisor of batch_size that results in
    // linear layer output buffers <= kMaxBufferSize, and a linear layer batch size
    // that is an integral multiple of 48.  Since the LSTM batch size is
    // already constrained to be an integral multiple of 48, this means the
    // batch splitting factor must be an exact divisor of the batch_size / 48.

    // If this target is smaller than the size required for 48 batch elements, then
    // that size is the best we can do.  The size here is attainable for fast and hac
    // models, but not sup.
    constexpr auto kMaxBufferSize = static_cast<int64_t>(1) << 29;
    const auto complete_linear_out_size =
            static_cast<int64_t>(m_out_chunk_size) * static_cast<int64_t>(m_batch_size) *
            static_cast<int64_t>(model_config.outsize) * sizeof(float);
    const int num_batch_pieces = m_batch_size / MTL_CORE_BATCH_SIZE;
    for (m_out_split = 1; m_out_split < num_batch_pieces; ++m_out_split) {
        if (num_batch_pieces % m_out_split == 0 &&
            complete_linear_out_size / m_out_split <= kMaxBufferSize) {
            break;
        }
    }
    auto piece_size = complete_linear_out_size / m_out_split;
    if (piece_size > kMaxBufferSize) {
        DORADO_LOG(debug, "Did not hit linear layer target output size {} - got {}", kMaxBufferSize,
                   piece_size);
    }
    DORADO_LOG(debug, "Linear layer split {}", m_out_split);
    // If we exited the loop above without breaking, then m_out_split = num_batch_pieces,
    // which satisfies the divisor criterion, and should mean small enough linear layer
    // output buffers, given other reasonable parameters.
    assert(num_batch_pieces % m_out_split == 0);
    assert(m_batch_size % m_out_split == 0);
    m_out_batch_size = m_batch_size / m_out_split;
    assert(m_out_batch_size % MTL_CORE_BATCH_SIZE == 0);

    m_model = std::make_unique<model::MetalCRFModelImpl>(model_config, m_in_chunk_size,
                                                         m_batch_size, m_out_split, m_device.get());
    m_model->load_state_dict(state_dict);
    m_model->eval();

    m_decode_complete_event = NS::TransferPtr(m_device->newSharedEvent());
    m_bwd_scan_cps = make_cps(m_device.get(), "backward_scan", {}, std::nullopt);
    m_fwd_scan_add_softmax_cps =
            make_cps(m_device.get(), "forward_scan_add_softmax", {}, std::nullopt);

    int T = m_out_chunk_size;
    int C = model_config.outsize;
    int Cs = m_states;

    m_scores_TNC.clear();
    m_posts_NTC.clear();
    m_bwd_NTC.clear();
    for (int i = 0; i < m_out_split; ++i) {
        m_scores_TNC.push_back(torch::empty({T, m_out_batch_size, C}, scores_dtype));
        // Unfortunately torch doesn't have Uint16, or we would use it.  We could offset,
        // or rely on undefined overflow behaviour, but for now we waste the sign bit.
        m_posts_NTC.push_back(torch::empty({m_out_batch_size, T + 1, Cs}, posts_dtype));
        m_bwd_NTC.push_back(torch::empty({m_out_batch_size, T + 1, Cs}));
    }
}

int MetalLSTMCaller::benchmark_batch_sizes(const BasecallModelConfig &model_config,
                                           const std::vector<at::Tensor> &state_dict,
                                           float memory_limit_fraction) {
    const int max_batch_size = get_max_safe_batch_size(memory_limit_fraction, model_config);

    // Subject to the above memory constraint, impose a minimum batch size
    // that will use 1/4 of GPU cores for LSTM execution.
    const int min_batch_size =
            std::min(MTL_CORE_BATCH_SIZE * get_mtl_device_core_count() / 4, max_batch_size);
    DORADO_LOG(trace, "min_batch_size {}", min_batch_size);

    std::set<int> test_batch_sizes{max_batch_size};

    // Add some batch sizes evenly distributed in between.
    const int kNumSmallerSizes = 16;
    const float test_size_increment = static_cast<float>(max_batch_size - min_batch_size) /
                                      static_cast<float>(kNumSmallerSizes);
    for (int i = 0; i <= kNumSmallerSizes; ++i) {
        const int test_batch_size =
                utils::pad_to(min_batch_size + static_cast<int>(i * test_size_increment),
                              static_cast<int>(MTL_CORE_BATCH_SIZE));
        test_batch_sizes.insert(test_batch_size);
    }

    // To speed up test runs, use a smaller chunk size.  This means we will not see
    // the true effect of memory thrashing, so we are relying on the memory limit
    // above to avoid that scenario.
    const int benchmark_chunk_size =
            std::min(model_config.basecaller.chunk_size(),
                     model_config.stride_inner() * 300 / model_config.scale_factor());

    // Iterate through batch size candidates to find the most efficient one.
    int best_batch_size = -1;
    long long best_us_per_batch_element = std::numeric_limits<long long>::max();
    for (int batch_size : test_batch_sizes) {
        DORADO_LOG(debug, "Trying batch size {}", batch_size);
        set_chunk_batch_size(model_config, state_dict, benchmark_chunk_size, batch_size);
        auto dummy_input = torch::empty(
                {batch_size, benchmark_chunk_size, model_config.num_features}, torch::kF16);
        const auto start_time = std::chrono::system_clock::now();
        auto *cb = m_model->forward_async(dummy_input, nullptr, 0, 0, m_scores_TNC);
        run_scan_kernels(cb, 0);
        const auto end_time = std::chrono::system_clock::now();
        const auto elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                        .count();
        const auto us_per_batch_element = elapsed_us / batch_size;
        DORADO_LOG(debug, "Batch {} us Batch element {} us", elapsed_us, us_per_batch_element);
        if (us_per_batch_element < best_us_per_batch_element) {
            best_us_per_batch_element = us_per_batch_element;
            best_batch_size = batch_size;
        }
    }
    assert(best_batch_size >= MTL_CORE_BATCH_SIZE);
    assert(best_batch_size % MTL_CORE_BATCH_SIZE == 0);
    return best_batch_size;
}

bool MetalLSTMCaller::run_scan_kernels(MTL::CommandBuffer *const cb, int try_count) {
    POINT_OF_INTEREST_SCOPE(MetalLSTMCaller, run_scan_kernels, "try_count=%i", try_count);

    // This stage is operating on the split outputs of the linear layer, so
    // the effective batch size is m_out_batch_size.
    std::vector<int32_t> scan_args_{m_out_chunk_size, m_out_batch_size, m_states};
    auto scan_args = create_vec_buffer(m_device.get(), scan_args_);
    name_mtl_object(scan_args, "scan_kernel_args");

    for (int i = 0; i < m_out_split; ++i) {
        // TODO: optimise grid size
        launch_kernel_no_wait(m_bwd_scan_cps.get(), cb,
                              {scan_args.get(), mtl_for_tensor(m_scores_TNC.at(i)),
                               mtl_for_tensor(m_bwd_NTC.at(i))},
                              {}, m_out_batch_size, m_states);

        launch_kernel_no_wait(m_fwd_scan_add_softmax_cps.get(), cb,
                              {scan_args.get(), mtl_for_tensor(m_scores_TNC.at(i)),
                               mtl_for_tensor(m_bwd_NTC.at(i)), mtl_for_tensor(m_posts_NTC.at(i))},
                              {}, m_out_batch_size, m_states);
    }
    return run_command_buffer("linear/scan/softmax", cb, try_count);
}

bool MetalLSTMCaller::call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) {
    std::lock_guard lock(inter_caller_mutex);

    // The linear layer should not execute until the previous batch has been decoded,
    // since the same buffers are used for successive batches' scores, fwd/bwd scans.
    MTL::CommandBuffer *const cb =
            m_model->forward_async(*task.input, m_decode_complete_event.get(),
                                   task.decode_complete_event_id - 1, try_count, m_scores_TNC);
    if (cb == nullptr) {
        return false;
    }

    return run_scan_kernels(cb, try_count);
}

DecodedData MetalLSTMCaller::decode(int chunk_idx) const {
    POINT_OF_INTEREST_SCOPE(MetalLSTMCaller, decode, "chunk_idx=%i", chunk_idx);

    // Model outputs are split across m_out_split buffers.
    assert(m_scores_TNC.size() == static_cast<size_t>(m_out_split));
    assert(m_bwd_NTC.size() == static_cast<size_t>(m_out_split));
    assert(m_posts_NTC.size() == static_cast<size_t>(m_out_split));
    const int out_buf_idx = chunk_idx / m_out_batch_size;
    const int buf_chunk_idx = chunk_idx % m_out_batch_size;

    return decode::beam_search_decode(
            // LSTM: m_scores_TNC[:, buf_chunk_idx, :] -> scores_TC
            m_scores_TNC.at(out_buf_idx).index({Slice(), buf_chunk_idx}),
            m_bwd_NTC.at(out_buf_idx)[buf_chunk_idx], m_posts_NTC.at(out_buf_idx)[buf_chunk_idx],
            m_decoder_options.beam_width, m_decoder_options.beam_cut, m_decoder_options.blank_score,
            m_decoder_options.q_shift, m_decoder_options.q_scale, m_score_scale);
}

}  // namespace dorado::basecall
