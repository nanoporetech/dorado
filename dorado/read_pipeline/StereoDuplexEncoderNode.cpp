#include "StereoDuplexEncoderNode.h"

#include "3rdparty/edlib/edlib/include/edlib.h"
#include "utils/duplex_utils.h"

#include <chrono>

using namespace std::chrono_literals;
using namespace torch::indexing;

namespace {
std::shared_ptr<dorado::Read> stereo_encode(std::shared_ptr<dorado::Read> template_read,
                                            std::shared_ptr<dorado::Read> complement_read) {
    // As a first step, let's return basespace calls, just as a sanity test and code-cleanup opportunity
    EdlibAlignConfig align_config = edlibDefaultAlignConfig();
    align_config.task = EDLIB_TASK_PATH;

    std::vector<char> complement_sequence_reverse_complement(complement_read->seq.begin(),
                                                             complement_read->seq.end());
    dorado::utils::reverse_complement(complement_sequence_reverse_complement);

    // Step 1 - let's align the two reads to one another and print out the score.
    EdlibAlignResult result =
            edlibAlign(template_read->seq.data(), template_read->seq.size(),
                       complement_sequence_reverse_complement.data(),
                       complement_sequence_reverse_complement.size(), align_config);

    std::cerr << result.editDistance << std::endl;
    // Step 2 - Prepare the encoding tensor

    int max_size = complement_sequence_reverse_complement.size() + template_read->seq.size();
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    int num_features = 13;
    auto tmp = torch::empty({num_features, max_size}, opts);

    // Step 3 - Move along the alignment, filling out the encodoing tensor

    // Step 4 - Assign the data to a new read,
    return template_read;
}
}  // namespace

namespace dorado {
// Let's make a stub which just consumes reads from its input queue and passes it to its output queue.
// Next step

void StereoDuplexEncoderNode::worker_thread() {
    int i = 0;

    while (true) {
        // Wait until we are provided with a read
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait_for(lock, 100ms, [this] { return !m_reads.empty(); });
        if (m_reads.empty()) {
            if (m_terminate) {
                // Termination flag is set and read input queue is empty, so terminate the worker
                return;
            } else {
                continue;
            }
        }

        std::shared_ptr<Read> read = m_reads.front();
        m_reads.pop_front();
        lock.unlock();

        bool read_is_template = false;
        bool read_is_complement = false;
        bool partner_found = false;
        std::string partner_id;

        // Check if read is a template with corresponding complement
        if (m_template_complement_map.find(read->read_id) != m_template_complement_map.end()) {
            read_is_template = true;
            partner_id = m_template_complement_map[read->read_id];
            partner_found = true;
            //std::cerr<< "Read is template" << std::endl;
        } else {
            if (m_complement_template_map.find(read->read_id) != m_complement_template_map.end()) {
                read_is_complement = true;
                partner_id = m_complement_template_map[read->read_id];
                partner_found = true;
                //std::cerr<< "Read is complement" << std::endl;
            }
        }

        if (partner_found) {
            i++;
            if (i % 100 == 0) {
                std::cerr << i << std::endl;
            }

            if (read_cache.find(partner_id) == read_cache.end()) {
                // Partner is not in the read cache
                read_cache[read->read_id] = read;
            } else {
                std::shared_ptr<Read> template_read;
                std::shared_ptr<Read> complement_read;

                auto partner_read_itr = read_cache.find(partner_id);
                auto partner_read = partner_read_itr->second;
                read_cache.erase(partner_read_itr);

                if (read_is_template) {
                    template_read = read;
                    complement_read = partner_read;
                } else {
                    complement_read = read;
                    template_read = partner_read;
                }

                std::shared_ptr<Read> stereo_encoded_read =
                        stereo_encode(template_read, complement_read);
                if (stereo_encoded_read->raw_data.numel() > 0) {
                    m_sink.push_read(stereo_encoded_read);  // Found a partner, so process it.
                }
            }
        }
    }
}

StereoDuplexEncoderNode::StereoDuplexEncoderNode(
        ReadSink &sink,
        std::map<std::string, std::string> template_complement_map)
        : ReadSink(1000), m_sink(sink), m_template_complement_map(template_complement_map) {
    // Set up teh complement_template_map

    for (auto key : template_complement_map) {
        m_complement_template_map[key.second] = key.first;
    }

    std::unique_ptr<std::thread> stereo_encoder_worker_thread =
            std::make_unique<std::thread>(&StereoDuplexEncoderNode::worker_thread, this);
    worker_threads.push_back(std::move(stereo_encoder_worker_thread));
}

StereoDuplexEncoderNode::~StereoDuplexEncoderNode() {
    terminate();
    m_cv.notify_one();
    for (auto &t : worker_threads) {
        t->join();
    }
}

}  // namespace dorado
