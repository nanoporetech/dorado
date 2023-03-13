#include "DuplexSplitNode.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

namespace {

using namespace dorado;

typedef DuplexSplitNode::PosRange PosRange;

//                           T  A     T        T  C     A     G        T     A  C
//std::vector<uint8_t> moves{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0};
std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t> moves) {
    std::vector<uint64_t> ans(moves.size(), 0);
    if (!moves.empty()) {
        ans[0] = moves[0];
    }
    for (size_t i = 1, n = moves.size(); i < n; i++) {
        ans[i] = ans[i-1] + moves[i];
    }
    return ans;
}

std::vector<std::pair<size_t, size_t>> detect_pore_signal(torch::Tensor pa_signal,
                                                          float threshold,
                                                          size_t cluster_dist) {
    std::vector<std::pair<size_t, size_t>> ans;
    auto pore_positions = torch::nonzero(pa_signal > threshold);
    //FIXME what type to use here?
    auto pore_a = pore_positions.accessor<int32_t, 1>();
    size_t start = size_t(-1);
    size_t end = size_t(-1);

    int i = 0;
    for (; i < pore_a.size(0); i++) {
        if (i == 0 || pore_a[i] > end + cluster_dist) {
            if (i > 0) {
                ans.push_back(std::pair{start, end});
            }
            start = pore_a[i];
        }
        end = pore_a[i] + 1;
    }
    if (i > 0) {
        ans.push_back(std::pair{start, end});
    }
    return ans;
}

bool check_rc_match(const std::string& seq, PosRange r1, PosRange r2) {
    //FIXME TODO
    return true;
}

std::shared_ptr<Read> subread(const Read& read, PosRange seq_range, PosRange signal_range) {
    //FIXME TODO
    //FIXME do we have end_reason access?

    /*
    new_read->raw_data = ...read->raw_data.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
    new_read->sample_rate = run_sample_rate;
    auto start_time_ms =
            run_acquisition_start_time_ms + ((read_data.start_sample * 1000) / run_sample_rate);
    auto start_time = get_string_timestamp_from_unix_time(start_time_ms);
    new_read->scaling = read_data.calibration_scale;
    new_read->offset = read_data.calibration_offset;
    new_read->read_id = std::move(read_id_str);
    new_read->num_trimmed_samples = 0;
    new_read->attributes.read_number = read_data.read_number;
    new_read->attributes.fast5_filename = std::filesystem::path(path.c_str()).filename().string();
    new_read->attributes.mux = read_data.well;
    new_read->attributes.channel_number = read_data.channel;
    new_read->attributes.start_time = start_time;
    new_read->run_id = run_info_data->protocol_run_id;
    new_read->scale = ...
    new_read->shift = ...
    new_read->seq = ...
    new_read->qstring = ...
    new_read->moves = ...
    */
    return std::make_shared<Read>();
}

}

namespace dorado {

//empty return means that there is no need to split
//std::vector<ReadRange>
std::vector<DuplexSplitNode::PosRange>
DuplexSplitNode::identify_splits(const Read& read) {
    spdlog::info("DSN: Searching for splits in read {}", read.read_id);
    std::vector<PosRange> interspace_regions;
    const auto move_sums = move_cum_sums(read.moves);
    assert(move_sums.back() == read.seq.length());

    //pA formula before scaling:
    //pA = read->scaling * (raw + read->offset);
    //pA formula after scaling:
    //pA = read->scale * raw + read->shift
    for (auto pore_level_region : detect_pore_signal(
                                    read.raw_data * read.scale + read.shift,
                                    m_settings.pore_thr,
                                    m_settings.pore_cl_dist)) {
        auto move_start = pore_level_region.first / read.model_stride;
        auto move_end = pore_level_region.second / read.model_stride;
        assert(move_end >= move_start);
        if (move_sums.at(move_start) == 0) {
            continue;
        }
        auto start_pos = move_sums.at(move_start) - 1;
        //NB. adding adapter length
        auto end_pos = move_sums.at(move_end);
        assert(end_pos > start_pos);
        if (start_pos < m_settings.flank_size ||
            end_pos + m_settings.adapter_length + m_settings.flank_size > read.seq.length()) {
            //TODO maybe handle this case with extra min_avail_sequence parameter
            continue;
        }

        //FIXME should we subtract a portion of adapter from the first region too?
        if (check_rc_match(read.seq,
                        {start_pos - m_settings.flank_size, start_pos},
                        {end_pos + m_settings.adapter_length,
                            end_pos + m_settings.adapter_length + m_settings.flank_size})) {
            interspace_regions.push_back(PosRange{start_pos, end_pos});
        }
    }

    return interspace_regions;
}

std::vector<Message> DuplexSplitNode::split(const Read& read,
                                            const std::vector<PosRange> &interspace_regions) {
    assert(!interspace_regions.empty());
    std::vector<Message> ans;
    ans.reserve(interspace_regions.size() + 1);

    const auto seq_to_sig_map = utils::moves_to_map(
            read.moves, read.model_stride, read.raw_data.size(0), read.seq.size() + 1);

    //TODO maybe simplify by adding begin/end stubs?
    uint64_t start_pos = 0;
    uint64_t signal_start = seq_to_sig_map[0];
    for (auto& r: interspace_regions) {
        //FIXME Don't forget to correctly process end_reasons!
        ans.push_back(subread(read, {start_pos, r.first}, {signal_start, seq_to_sig_map[r.first]}));
        start_pos = r.second;
        signal_start = seq_to_sig_map[r.second];
    }
    ans.push_back(subread(read, {start_pos, read.seq.size()}, {signal_start, read.raw_data.size(0)}));
    return ans;
}

void DuplexSplitNode::worker_thread() {
    spdlog::info("DSN: Hello from worker thread");
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        auto ranges = identify_splits(*read);
        if (ranges.empty()) {
            m_sink.push_message(read);
        } else {
            for (auto m : split(*read, ranges)) {
                m_sink.push_message(std::move(m));
            }
        }
    }
}

DuplexSplitNode::DuplexSplitNode(MessageSink& sink, DuplexSplitSettings settings,
                                int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads), m_sink(sink),
            m_settings(settings), m_num_worker_threads(num_worker_threads) {
    for (int i = 0; i < m_num_worker_threads; i++) {
        worker_threads.push_back(std::make_unique<std::thread>(&DuplexSplitNode::worker_thread, this));
    }
}

DuplexSplitNode::~DuplexSplitNode() {
    terminate();

    // Wait for all the Node's worker threads to terminate
    for (auto& t : worker_threads) {
        t->join();
    }

    // Notify the sink that the Node has terminated
    m_sink.terminate();
}

}