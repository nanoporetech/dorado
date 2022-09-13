#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "MetalCRFModel.h"

#include "../decode/beam_search.h"
#include "../utils/metal_utils.h"
#include "../utils/module_utils.h"
#include "../utils/tensor_utils.h"

#include <math.h>
#include <toml.hpp>
#include <torch/torch.h>
#include <vector>

using namespace std::chrono_literals;
using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;

typedef uint16_t ftype;

struct MetalLinearTanhImpl : Module {
    MetalLinearTanhImpl(int insize, int outsize) {
        auto weight = torch::empty({outsize, insize});
        auto bias = torch::empty({outsize});
        register_parameter("weight", weight, false);
        register_parameter("bias", bias, false);
    }
};

TORCH_MODULE(MetalLinearTanh);

struct MetalConv1dImpl : Module {
    MetalConv1dImpl(int layer,
                    int insize_,
                    int outsize_,
                    int k_,
                    int stride,
                    int chunk_size,
                    int batch_size,
                    MTL::Device *device)
            : insize(insize_), outsize(outsize_), k(k_) {
        assert(layer >= 1 && layer <= 3);

        if (layer == 1) {
            assert(outsize == 4);
            mat_weights = create_buffer(device, 49 * 8 * sizeof(ftype));
        } else if (layer == 2) {
            assert(outsize == 16);
            mat_weights = create_buffer(device, 29 * 16 * sizeof(ftype));
        } else {
            mat_weights = create_buffer(device, (k * insize + 1) * outsize * sizeof(ftype));
        }
        int32_t args_[] = {insize, k, outsize, stride, k / 2, chunk_size, batch_size};
        args = create_buffer(device, args_, 7);

        auto weight = torch::empty({outsize, insize, k});
        auto bias = torch::empty({outsize});

        register_parameter("weight", weight, false);
        register_parameter("bias", bias, false);

        conv_cps = make_cps(device, "conv" + std::to_string(layer) + "_simd");
        weights_cps = make_cps(device, "conv" + std::to_string(layer) + "_simd_reorder_weights");

        kernel_simd_groups = layer == 3 ? 4 : 16;
        kernel_thread_groups = get_mtl_device_core_count() * 4;
    }

    void run(MTL::CommandQueue *command_queue, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        std::vector<MTL::Buffer *> buffers{args, mat_in, mat_weights, mat_out};
        launch_kernel(conv_cps, command_queue, buffers, kernel_thread_groups,
                      kernel_simd_groups * 32);
    }

    void run(MTL::CommandBuffer *command_buffer, MTL::Buffer *mat_in, MTL::Buffer *mat_out) {
        std::vector<MTL::Buffer *> buffers{args, mat_in, mat_weights, mat_out};
        launch_kernel_no_wait(conv_cps, command_buffer, buffers, kernel_thread_groups,
                              kernel_simd_groups * 32);
    }

    void load_weights(MTL::CommandQueue *command_queue) {
        auto params = named_parameters();
        auto t_w = *params.find("weight");
        auto t_b = *params.find("bias");
        t_w = t_w.permute({2, 1, 0}).contiguous();
        t_w = t_w.reshape({insize * k, outsize});
        t_w = torch::concat({t_w.flatten(0, -1), t_b}).contiguous();
        launch_kernel(weights_cps, command_queue, {args, mtl_for_tensor(t_w), mat_weights}, 1, 1);
    }

    MTL::Buffer *args, *mat_weights;
    MTL::ComputePipelineState *conv_cps, *weights_cps;
    int kernel_simd_groups, kernel_thread_groups;
    int insize, outsize, k;
};

TORCH_MODULE(MetalConv1d);

struct MetalLSTMImpl : Module {
    MetalLSTMImpl(int layer_size, bool reverse_, MTL::Device *device) : reverse(reverse_) {
        auto weight_ih = torch::empty({layer_size * 4, layer_size});
        auto weight_hh = torch::empty({layer_size * 4, layer_size});
        auto bias_ih = torch::empty({layer_size * 4});
        auto bias_hh = torch::empty({layer_size * 4});

        register_parameter("weight_ih", weight_ih, false);
        register_parameter("weight_hh", weight_hh, false);
        register_parameter("bias_ih", bias_ih, false);
        register_parameter("bias_hh", bias_hh, false);

        mat_weights = create_buffer(device,
                                    (size_t)(layer_size * 2 + 1) * layer_size * 4 * sizeof(ftype));
    }

    MTL::Buffer *mat_weights;
    bool reverse;
};

TORCH_MODULE(MetalLSTM);

struct MetalBlockImpl : Module {
    MetalBlockImpl(int chunk_size_,
                   int batch_size_,
                   int stride_,
                   int layer_size_,
                   int out_size_,
                   int out_split_,
                   MTL::Device *device_)
            : device(device_),
              in_chunk_size(chunk_size_),
              stride(stride_),
              batch_size(batch_size_),
              layer_size(layer_size_),
              out_size(out_size_) {
        command_queue = device->newCommandQueue();

        constexpr int tile_size = 8;

        lstm_chunk_size = in_chunk_size / stride;

        // args for forward LSTM
        int32_t args_lstm_[] = {layer_size, 0, batch_size / tile_size, lstm_chunk_size, out_size};
        args_lstm[0] = create_buffer(device, args_lstm_, 5);
        // args for reverse LSTM
        args_lstm_[1] = 1;
        args_lstm[1] = create_buffer(device, args_lstm_, 5);

        // args for conversion to half
        int32_t args_to_half_[] = {in_chunk_size * batch_size};
        args_to_half = create_buffer(device, args_to_half_, 1);

        // args for linear layer
        // Each output buffer requires a distinct input offset, so we must have a separate args buffer.
        args_linear.resize(out_split_);
        for (int i = 0; i < out_split_; ++i) {
            const int32_t in_batch_tiles = batch_size / tile_size;
            const int32_t out_batch_tiles = (batch_size / out_split_) / tile_size;
            const int32_t in_batch_tile_offset = out_batch_tiles * i;
            int32_t args_linear_[] = {in_batch_tiles, in_batch_tile_offset, out_batch_tiles,
                                      lstm_chunk_size, out_size};
            args_linear.at(i) = create_buffer(device, args_linear_, 5);
        }

        switch (layer_size) {
        case 128:
            kernel_simd_groups = 16;
            break;
        case 192:
            kernel_simd_groups = 12;
            break;
        case 256:
            kernel_simd_groups = 32;
            break;
        case 384:
            kernel_simd_groups = 24;
            break;
        case 512:
            kernel_simd_groups = 32;
            break;
        default:
            kernel_simd_groups = 16;
        }
        kernel_thread_groups = get_mtl_device_core_count();

        fn[0] = "lstm_simd_" + std::to_string(layer_size) + "_fwd_" +
                std::to_string(kernel_simd_groups);
        fn[1] = "lstm_simd_" + std::to_string(layer_size) + "_rev_" +
                std::to_string(kernel_simd_groups);
        std::string linear_fn = "linear_tanh_simd_" + std::to_string(layer_size) + "_fwd_" +
                                std::to_string(kernel_simd_groups);
        lstm_cps[0] = make_cps(device, fn[0]);
        lstm_cps[1] = make_cps(device, fn[1]);
        linear_tanh_cps = make_cps(device, linear_fn);
        to_half_cps = make_cps(device, "float_to_half");
        reorder_input_cps = make_cps(device, "reorder_input");
        reorder_output_cps = make_cps(device, "reorder_output");
        reorder_weights_cps = make_cps(device, "reorder_weights");

        // TODO: we can reuse some of these matrices
        mat_transfer = create_buffer(
                device, (size_t)batch_size * lstm_chunk_size * out_size * sizeof(float));
        mat_transfer_ftype =
                (sizeof(ftype) == 4)
                        ? mat_transfer
                        : create_buffer(device, (size_t)batch_size * in_chunk_size * sizeof(ftype));
        mat_working_mem = create_buffer(
                device, size_t(lstm_chunk_size + 2) * batch_size * layer_size * sizeof(ftype));
        mat_state = create_buffer(device, batch_size * layer_size * sizeof(ftype));
        mat_temp_result = create_buffer(device, batch_size * layer_size * 4 * sizeof(ftype));

        conv1 = register_module("conv1",
                                MetalConv1d(1, 1, 4, 5, 1, in_chunk_size, batch_size, device));
        conv2 = register_module("conv2",
                                MetalConv1d(2, 4, 16, 5, 1, in_chunk_size, batch_size, device));
        conv3 = register_module("conv3", MetalConv1d(3, 16, layer_size, 19, stride, in_chunk_size,
                                                     batch_size, device));
        rnn1 = register_module("rnn_1", MetalLSTM(layer_size, true, device));
        rnn2 = register_module("rnn_2", MetalLSTM(layer_size, false, device));
        rnn3 = register_module("rnn_3", MetalLSTM(layer_size, true, device));
        rnn4 = register_module("rnn_4", MetalLSTM(layer_size, false, device));
        rnn5 = register_module("rnn_5", MetalLSTM(layer_size, true, device));
        linear = register_module("linear", MetalLinearTanh(layer_size, out_size));
    }

    void load_weights() {
        conv1->load_weights(command_queue);
        conv2->load_weights(command_queue);
        conv3->load_weights(command_queue);

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            auto params = rnn->named_parameters();

            auto t_w = *params.find("weight_ih");
            auto t_u = *params.find("weight_hh");
            auto t_b = *params.find("bias_ih");

            // reorder from IFGO to GIFO (2, 0, 1, 3)
            t_w = t_w.reshape({4, layer_size, layer_size}).transpose(1, 2);
            t_w = torch::concat({t_w[2], t_w[0], t_w[1], t_w[3]}, 1);

            t_u = t_u.reshape({4, layer_size, layer_size}).transpose(1, 2);
            t_u = torch::concat({t_u[2], t_u[0], t_u[1], t_u[3]}, 1);

            t_b = t_b.reshape({4, layer_size});
            t_b = torch::concat({t_b[2], t_b[0], t_b[1], t_b[3]});

            t_w = t_w.flatten(0, -1).contiguous();
            t_u = t_u.flatten(0, -1).contiguous();
            t_b = t_b.flatten(0, -1).contiguous();

            std::vector<MTL::Buffer *> buffers{args_lstm[rnn->reverse], mtl_for_tensor(t_w),
                                               mtl_for_tensor(t_u), mtl_for_tensor(t_b),
                                               rnn->mat_weights};
            launch_kernel(reorder_weights_cps, command_queue, buffers, kernel_thread_groups, 256);
        }

        auto params = linear->named_parameters();
        auto t_w = *params.find("weight");
        auto t_b = *params.find("bias");
        t_w = torch::concat({t_w.transpose(1, 0).contiguous().flatten(0, -1), t_b}).contiguous();

        if (sizeof(ftype) != sizeof(float)) {
            int numel = int(t_w.numel());
            MTL::Buffer *args = create_buffer(device, &numel, 1);
            mat_linear_weights = create_buffer(device, numel * sizeof(ftype));
            launch_kernel(to_half_cps, command_queue,
                          {args, mtl_for_tensor(t_w), mat_linear_weights}, kernel_thread_groups,
                          256);
        } else {
            mat_linear_weights = extract_mtl_from_tensor(t_w);
        }
    }

    // Executes the model, with the linear layer held off by linear_hold_off, if non-NULL.
    MTL::CommandBuffer *forward_async(torch::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      int linear_hold_off_id,
                                      std::vector<torch::Tensor> &out) {
        auto command_buffer = command_queue->commandBuffer();

        if (sizeof(ftype) == 2) {
            launch_kernel_no_wait(to_half_cps, command_buffer,
                                  {args_to_half, mtl_for_tensor(in), mat_transfer_ftype},
                                  kernel_thread_groups, 256);
            conv1->run(command_buffer, mat_transfer_ftype, mat_working_mem);
        } else {
            conv1->run(command_buffer, mtl_for_tensor(in), mat_working_mem);
        }
        conv2->run(command_buffer, mat_working_mem, mat_transfer);
        conv3->run(command_buffer, mat_transfer, mat_working_mem);

        for (auto &rnn : {rnn1, rnn2, rnn3, rnn4, rnn5}) {
            std::vector<MTL::Buffer *> buffers{args_lstm[rnn->reverse], mat_working_mem,
                                               rnn->mat_weights, mat_state, mat_temp_result};
            launch_kernel_no_wait(lstm_cps[rnn->reverse], command_buffer, buffers,
                                  kernel_thread_groups, kernel_simd_groups * 32);
        }

        // The output buffers of conv/LSTM layers are not used by the decoding, so
        // can be overwritten by subsequent batches as soon as they have been consumed by
        // the linear layer.  The output of the linear layer must be protected until
        // it has been decoded.
        if (linear_hold_off_event != nullptr) {
            command_buffer->encodeWait(linear_hold_off_event, linear_hold_off_id);
        }

        // The output of the linear layer is split into multiple buffers, each generated
        // by a separate kernel launch.
        for (int i = 0; i < out.size(); ++i) {
            Buffer* const args_buffer = args_linear.at(i);
            Buffer* const out_buffer = mtl_for_tensor(out.at(i));
            launch_kernel_no_wait(
                    linear_tanh_cps, command_buffer,
                    {args_buffer, mat_working_mem, mat_linear_weights, out_buffer},
                    kernel_thread_groups, kernel_simd_groups * 32);
        }
        return command_buffer;
    }

    torch::Tensor forward(torch::Tensor in) {
        std::vector<torch::Tensor> out { torch::empty({lstm_chunk_size, batch_size, out_size}) };
        // TODO: find a more robust way of dealing with Metal kernel launch issues
        for (int try_count = 0; try_count < 5; ++try_count) {
            MTL::CommandBuffer *command_buffer = forward_async(in, nullptr, 0, out);
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
            if (command_buffer->status() == MTL::CommandBufferStatusCompleted) {
                break;
            }
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(20ms);
        }
        return out.at(0);
    }

    MTL::Device *device;
    MTL::CommandQueue *command_queue;
    std::string fn[2];
    MTL::ComputePipelineState *reorder_weights_cps, *reorder_input_cps, *reorder_output_cps,
            *lstm_cps[2], *to_half_cps, *linear_tanh_cps;
    MTL::Buffer *mat_transfer, *mat_transfer_ftype, *mat_working_mem, *mat_state, *mat_temp_result,
    *mat_linear_weights, *args_lstm[2], *args_to_half;
    std::vector<MTL::Buffer*> args_linear;
    int in_chunk_size, lstm_chunk_size, stride, batch_size, layer_size, out_size,
            kernel_thread_groups, kernel_simd_groups;
    MetalLSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
    MetalConv1d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    MetalLinearTanh linear{nullptr};
};

TORCH_MODULE(MetalBlock);

struct MetalModelImpl : Module {
    MetalModelImpl(int size,
                   int outsize,
                   int stride,
                   int chunk_size,
                   int batch_size,
                   int out_split,
                   MTL::Device *device) {
        mtl_block = register_module(
                "mtl_block", MetalBlock(chunk_size, batch_size, stride, size, outsize, out_split, device));
    }

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        ::utils::load_state_dict(*this, weights);
        mtl_block->load_weights();
    }

    torch::Tensor forward(torch::Tensor x) { return mtl_block->forward(x); }

    MTL::CommandBuffer *forward_async(torch::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      int linear_hold_off_id,
                                      std::vector<torch::Tensor> &out) {
        return mtl_block->forward_async(in, linear_hold_off_event, linear_hold_off_id, out);
    }

    MetalBlock mtl_block{nullptr};
};

TORCH_MODULE(MetalModel);

class MetalCaller {
public:
    MetalCaller(const std::filesystem::path &model_path, int chunk_size, int batch_size) {
        // LSTM kernels, which run with the full supplied batch size, require that the batch size
        // be an integral multiple of 48.
        assert(batch_size % 48 == 0);
        auto config = toml::parse(model_path / "config.toml");
        const auto &qscore = toml::find(config, "qscore");
        const auto qbias = toml::find<float>(qscore, "bias");
        const auto qscale = toml::find<float>(qscore, "scale");

        m_device = get_mtl_device();

        m_decoder_options = DecoderOptions();
        m_decoder_options.q_shift = qbias;
        m_decoder_options.q_scale = qscale;

        const auto &encoder = toml::find(config, "encoder");
        const auto scale = toml::find<float>(encoder, "scale");
        const auto stride = toml::find<int>(encoder, "stride");
        const auto insize = toml::find<int>(encoder, "features");
        const auto blank_score = toml::find<float>(encoder, "blank_score");

        const auto &global_norm = toml::find(config, "global_norm");
        const auto state_len = toml::find<int>(global_norm, "state_len");

        constexpr int n_base = 4;
        constexpr int num_transitions = 5;

        m_states = pow(n_base, state_len);
        m_batch_size = batch_size;
        int outsize = m_states * num_transitions;
        
        m_model_stride = static_cast<size_t>(stride);
        // adjust chunk size to a multiple of the stride
        chunk_size -= chunk_size % stride;

        // Chunk size after decimation via convolution stride.
        m_out_chunk_size = chunk_size / stride;

        auto state_dict = load_weights(model_path);

        auto lw = state_dict[state_dict.size() - 2];
        auto lb = state_dict[state_dict.size() - 1];

        state_dict[state_dict.size() - 2] =
                F::pad(lw.view({m_states, 4, insize}), F::PadFuncOptions({0, 0, 1, 0}).value(0.0))
                        .view({outsize, insize});

        state_dict[state_dict.size() - 1] =
                F::pad(lb.view({m_states, 4}),
                       F::PadFuncOptions({1, 0}).value(atanh(blank_score / scale)))
                        .view({outsize});

        // Buffers visible to the CPU and GPU have a 4GB size limit, and the linear layer
        // output buffer hits this limit with batch sizes larger than 384 with typical
        // chunk sizes.  At the same time, the LSTM layer performance benefits from large batch sizes.
        // We therefore run the linear layer via 1 or more kernel runs, each with an output buffer
        // with a size <= 4GB, with a reduced batch size.
        // The linear layer kernel requires a batch size that is an integral multiple of 48.
        // As things stand, we need an exactly even split of batch elements in the linear
        // layer output buffers (this could be relaxed).
        // We therefore need the smallest divisor of batch_size that results in
        // linear layer output buffers < 4GB, and a linear layer batch size that is an
        // integral multiple of 48.  Since the LSTM batch size is already constrained to be an integral multiple of
        // 48, this means the batch splitting factor must be an exact divisor of
        // the batch_size / 48.
        constexpr auto kMaxBufferSize = static_cast<int64_t>(1) << 32;
        const auto complete_linear_out_size =
            static_cast<int64_t>(m_out_chunk_size) * static_cast<int64_t>(batch_size) * static_cast<int64_t>(outsize) * sizeof(float);
        const int num_batch_pieces = batch_size / 48;
        for (m_out_split = 1; m_out_split < num_batch_pieces; ++m_out_split) {
            if (num_batch_pieces % m_out_split == 0 &&
                complete_linear_out_size / m_out_split < kMaxBufferSize)
                break;
        }
        // If we exited the loop above without breaking, then m_out_split = num_batch_pieces,
        // which satisfies the divisor criterion, and should mean small enough linear layer
        // output buffers, given other reasonable parameters.
        assert(num_batch_pieces % m_out_split == 0);
        assert(complete_linear_out_size / m_out_split < kMaxBufferSize);
        assert(batch_size % m_out_split == 0);
        m_out_batch_size = batch_size / m_out_split;
        assert(m_out_batch_size % 48 == 0);

        m_model = MetalModel(insize, outsize, stride, chunk_size, batch_size, m_out_split, m_device);
        m_model->load_state_dict(state_dict);
        m_model->eval();

        m_command_queue = m_device->newCommandQueue();
        m_mtl_event = m_device->newSharedEvent();
        m_scan_cps = make_cps(m_device, "scan");
        m_add_softmax_cps = make_cps(m_device, "add_softmax");

        m_metal_thread.reset(new std::thread(&MetalCaller::metal_thread_fn, this));

        int num_decode_threads = std::max(1, get_apple_cpu_perf_core_count() - 1);
        m_decode_threads.reserve(num_decode_threads);
        for (int i = 0; i < num_decode_threads; ++i) {
            m_decode_threads.emplace_back(new std::thread(&MetalCaller::decode_thread_fn, this, i));
        }

        int T = m_out_chunk_size;
        int C = outsize;
        int Cs = m_states;

        int y = pow(n_base, state_len);

        m_scan_idx[0][0] = torch::arange(C, torch::kInt32).contiguous();
        auto t1 = torch::arange(y).index({torch::indexing::Slice(), torch::indexing::None});
        auto t2 = torch::arange(y).repeat_interleave(n_base).reshape({n_base, -1}).t();
        m_scan_idx[0][1] = torch::cat({t1, t2}, 1).to(torch::kInt32).contiguous();

        auto idx_sizes = m_scan_idx[0][1].sizes();
        m_scan_idx[1][0] = m_scan_idx[0][1]
                                   .flatten()
                                   .argsort()
                                   .reshape(idx_sizes)
                                   .to(torch::kInt32)
                                   .contiguous();
        m_scan_idx[1][1] = torch::div(m_scan_idx[1][0], num_transitions, "floor");

        for (int i = 0; i < m_out_split; ++i) {
            m_scores.push_back(torch::empty({T, m_out_batch_size, C}));
            m_posts.push_back(torch::empty({m_out_batch_size, T + 1, Cs}));
            m_bwd.push_back(torch::empty({m_out_batch_size, T + 1, Cs}));
        }
    }

    ~MetalCaller() {
        std::unique_lock<std::mutex> input_lock(m_input_lock);
        m_terminate = true;
        input_lock.unlock();
        m_input_cv.notify_one();
        m_decode_cv.notify_all();

        m_metal_thread->join();
        for (auto &thr : m_decode_threads) {
            thr->join();
        }
    }

    std::vector<torch::Tensor> load_weights(const std::filesystem::path &dir) {
        auto tensors = std::vector<std::string>{

                "0.conv.weight.tensor",      "0.conv.bias.tensor",

                "1.conv.weight.tensor",      "1.conv.bias.tensor",

                "2.conv.weight.tensor",      "2.conv.bias.tensor",

                "4.rnn.weight_ih_l0.tensor", "4.rnn.weight_hh_l0.tensor",
                "4.rnn.bias_ih_l0.tensor",   "4.rnn.bias_hh_l0.tensor",

                "5.rnn.weight_ih_l0.tensor", "5.rnn.weight_hh_l0.tensor",
                "5.rnn.bias_ih_l0.tensor",   "5.rnn.bias_hh_l0.tensor",

                "6.rnn.weight_ih_l0.tensor", "6.rnn.weight_hh_l0.tensor",
                "6.rnn.bias_ih_l0.tensor",   "6.rnn.bias_hh_l0.tensor",

                "7.rnn.weight_ih_l0.tensor", "7.rnn.weight_hh_l0.tensor",
                "7.rnn.bias_ih_l0.tensor",   "7.rnn.bias_hh_l0.tensor",

                "8.rnn.weight_ih_l0.tensor", "8.rnn.weight_hh_l0.tensor",
                "8.rnn.bias_ih_l0.tensor",   "8.rnn.bias_hh_l0.tensor",

                "9.linear.weight.tensor",    "9.linear.bias.tensor"};

        return ::utils::load_tensors(dir, tensors);
    }

    struct NNTask {
        NNTask(torch::Tensor *input_, int num_chunks_, std::vector<DecodedChunk> *out_chunks_)
                : input(input_), out_chunks(out_chunks_), num_chunks(num_chunks_) {
            static int run = 0;
            run_id = run++;
        }

        torch::Tensor *input;
        std::mutex mut;
        std::condition_variable cv;
        bool ready{false};
        std::vector<DecodedChunk> *out_chunks;
        int num_chunks;
        int decode_chunks_started{0};
        int decode_chunks_finished{0};
        int run_id;
    };

    void call_chunks(torch::Tensor &input, int num_chunks, std::vector<DecodedChunk> &out_chunks) {
        if (num_chunks == 0) {
            return;
        }

        NNTask task(&input, num_chunks, &out_chunks);
        {
            std::lock_guard<std::mutex> lock(m_input_lock);
            m_input_queue.push_front(&task);
        }
        m_input_cv.notify_one();

        std::unique_lock lock(task.mut);
        while (task.decode_chunks_finished != num_chunks) {
            task.cv.wait(lock);
        }
    }

    void metal_thread_fn() {
        while (true) {
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            while (m_input_queue.empty() && !m_terminate) {
                m_input_cv.wait_for(input_lock, 100ms);
            }
            // TODO: finish work before terminating?
            if (m_terminate) {
                return;
            }

            NNTask *task = m_input_queue.back();
            m_input_queue.pop_back();
            input_lock.unlock();

            // TODO: find a more robust way of dealing with Metal kernel launch issues
            for (int try_count = 0; try_count < 5; ++try_count) {
                // The linear layer should not execute until the previous batch has been decoded,
                // since the same buffers are used for successive batches' scores, fwd/bwd scans.
                MTL::SharedEvent *const linear_hold_off =
                        (task->run_id != 0) ? m_mtl_event : nullptr;
                MTL::CommandBuffer *const cb = m_model->forward_async(*task->input, linear_hold_off,
                                                                      task->run_id - 1, m_scores);

                // The same buffer is used for the forward scan results and the output of
                // m_add_softmax_cps.
                auto &fwd = m_posts;
                // This stage is operating on the split outputs of the linear layer, so
                // the effective batch size is m_out_batch_size.
                int32_t scan_args_[] = {m_out_chunk_size, m_out_batch_size, m_states,
                                        1};  // T, N, C, dir
                auto args_fwd = create_buffer(m_device, scan_args_, 4);
                scan_args_[3] = -1;
                auto args_bwd = create_buffer(m_device, scan_args_, 4);

                for (int i = 0; i < m_out_split; ++i) {
                    // TODO: optimise grid size
                    launch_kernel_no_wait(
                            m_scan_cps, cb,
                            {args_fwd, mtl_for_tensor(m_scores.at(i)), mtl_for_tensor(fwd.at(i)),
                             mtl_for_tensor(m_scan_idx[0][0]), mtl_for_tensor(m_scan_idx[0][1])},
                            m_out_batch_size, m_states);
                    launch_kernel_no_wait(
                            m_scan_cps, cb,
                            {args_bwd, mtl_for_tensor(m_scores.at(i)), mtl_for_tensor(m_bwd.at(i)),
                             mtl_for_tensor(m_scan_idx[1][0]), mtl_for_tensor(m_scan_idx[1][1])},
                            m_out_batch_size, m_states);
                    launch_kernel_no_wait(m_add_softmax_cps, cb,
                                          {args_fwd, mtl_for_tensor(fwd.at(i)), mtl_for_tensor(m_bwd.at(i))},
                                          m_out_batch_size, m_states);
                }

                cb->commit();
                cb->waitUntilCompleted();

                auto status = cb->status();
                if (status == MTL::CommandBufferStatusCompleted) {
                    break;
                }
                std::cerr << "Metal command buffer execution failed: " << status << ", try #"
                          << try_count << std::endl;
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(20ms);
            }

            // Pass task on to decode threads
            std::unique_lock<std::mutex> decode_lock(m_decode_lock);
            m_decode_queue.push_front(task);
            decode_lock.unlock();
            m_decode_cv.notify_all();
        }
    }

    void decode_thread_fn(int thread_id) {
        while (true) {
            std::unique_lock<std::mutex> decode_lock(m_decode_lock);
            while (m_decode_queue.empty() && !m_terminate) {
                m_decode_cv.wait_for(decode_lock, 100ms);
            }
            // TODO: finish work before terminating?
            if (m_terminate) {
                return;
            }
            NNTask *task = m_decode_queue.back();
            int chunk_idx = task->decode_chunks_started++;
            // If all chunks have been picked up for decoding, remove task from queue
            if (chunk_idx == task->num_chunks - 1) {
                m_decode_queue.pop_back();
            }
            decode_lock.unlock();

            // Model outputs are split across m_out_split buffers.
            assert(m_scores.size() == m_out_split);
            assert(m_bwd.size() == m_out_split);
            assert(m_posts.size() == m_out_split);
            const int out_buf_idx = chunk_idx / m_out_batch_size;
            const int buf_chunk_idx = chunk_idx % m_out_batch_size;

            auto decode_result =
                    beam_search_decode(m_scores.at(out_buf_idx).index({Slice(), buf_chunk_idx}),
                                       m_bwd.at(out_buf_idx)[buf_chunk_idx],
                                       m_posts.at(out_buf_idx)[buf_chunk_idx], m_decoder_options.beam_cut,
                                       m_decoder_options.blank_score, m_decoder_options.q_shift,
                                       m_decoder_options.q_scale, m_decoder_options.temperature);
            (*task->out_chunks)[chunk_idx] = DecodedChunk{
                    std::get<0>(decode_result),
                    std::get<1>(decode_result),
                    std::get<2>(decode_result),
            };

            // Wake the waiting thread which called `call_chunks()` if we're done decoding
            std::unique_lock<std::mutex> task_lock(task->mut);
            bool done = ++(task->decode_chunks_finished) == task->num_chunks;
            task_lock.unlock();
            if (done) {
                m_mtl_event->setSignaledValue(task->run_id);
                task->cv.notify_one();
            }
        }
    }

    bool m_terminate{false};
    std::deque<NNTask *> m_input_queue;
    std::deque<NNTask *> m_decode_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_metal_thread;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    std::vector<std::unique_ptr<std::thread>> m_decode_threads;
    DecoderOptions m_decoder_options;
    MetalModel m_model{nullptr};
    MTL::Device *m_device;
    MTL::CommandQueue *m_command_queue;
    MTL::ComputePipelineState *m_scan_cps, *m_add_softmax_cps;
    MTL::SharedEvent *m_mtl_event;
    torch::Tensor m_scan_idx[2][2];
    std::vector<torch::Tensor> m_scores, m_posts, m_bwd;
    int m_out_chunk_size, m_batch_size, m_states, m_model_stride;
    // Number of pieces the linear output is split into, for reasons of
    // buffer size constraints.
    int m_out_split;
    int m_out_batch_size;
};

std::shared_ptr<MetalCaller> create_metal_caller(const std::filesystem::path &model_path,
                                                 int chunk_size,
                                                 int batch_size) {
    return std::make_shared<MetalCaller>(model_path, chunk_size, batch_size);
}

MetalModelRunner::MetalModelRunner(std::shared_ptr<MetalCaller> caller,
                                   int chunk_size,
                                   int batch_size)
        : m_caller(caller) {
    m_input = torch::empty({batch_size, 1, chunk_size},
                           torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU));
}

void MetalModelRunner::accept_chunk(int chunk_idx, at::Tensor slice) {
    m_input.index_put_({chunk_idx, 0}, slice);
}

std::vector<DecodedChunk> MetalModelRunner::call_chunks(int num_chunks) {
    std::vector<DecodedChunk> out_chunks(num_chunks);
    m_caller->call_chunks(m_input, num_chunks, out_chunks);
    return out_chunks;
}

size_t MetalModelRunner::model_stride() const { return m_caller->m_model_stride; }
