#pragma once
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Pod5FileReader;

namespace dorado {

class MessageSink;
struct ReadGroup;

constexpr size_t POD5_READ_ID_SIZE = 16;
using ReadID = std::array<uint8_t, POD5_READ_ID_SIZE>;
typedef std::map<int, std::vector<ReadID>> channel_to_read_id_t;

struct Pod5Destructor {
    void operator()(Pod5FileReader*);
};
using Pod5Ptr = std::unique_ptr<Pod5FileReader, Pod5Destructor>;

class DataLoader {
public:
    enum ReadOrder {
        UNRESTRICTED,
        BY_CHANNEL,
    };

    DataLoader(MessageSink& read_sink,
               const std::string& device,
               size_t num_worker_threads,
               size_t max_reads = 0,
               std::optional<std::unordered_set<std::string>> read_list = std::nullopt);
    ~DataLoader() = default;
    void load_reads(const std::string& path,
                    bool recursive_file_loading = false,
                    ReadOrder traversal_order = UNRESTRICTED);

    static std::unordered_map<std::string, ReadGroup> load_read_groups(
            std::string data_path,
            std::string model_path,
            bool recursive_file_loading = false);

    static int get_num_reads(
            std::string data_path,
            std::optional<std::unordered_set<std::string>> read_list = std::nullopt,
            bool recursive_file_loading = false);

    static uint16_t get_sample_rate(std::string data_path, bool recursive_file_loading = false);

private:
    void load_fast5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                               const std::vector<ReadID>& read_ids);
    void load_read_channels(std::string data_path, bool recursive_file_loading = false);
    MessageSink& m_read_sink;  // Where should the loaded reads go?
    size_t m_loaded_read_count{0};
    std::string m_device;
    size_t m_num_worker_threads{1};
    size_t m_max_reads{0};
    std::optional<std::unordered_set<std::string>> m_allowed_read_ids;

    std::unordered_map<std::string, channel_to_read_id_t> m_file_channel_read_order_map;
    int m_max_channel{0};

    std::unordered_map<std::string, Pod5Ptr> m_file_handles;
};

}  // namespace dorado
