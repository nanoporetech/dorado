#pragma once
#include "utils/stats.h"
#include "utils/types.h"

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

class Pipeline;
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
    DataLoader(Pipeline& pipeline,
               const std::string& device,
               size_t num_worker_threads,
               size_t max_reads = 0,
               std::optional<std::unordered_set<std::string>> read_list = std::nullopt,
               std::unordered_set<std::string> read_ignore_list = {});
    ~DataLoader() = default;
    void load_reads(const std::string& path,
                    bool recursive_file_loading = false,
                    ReadOrder traversal_order = ReadOrder::UNRESTRICTED);

    static std::unordered_map<std::string, ReadGroup> load_read_groups(
            std::string data_path,
            std::string model_path,
            bool recursive_file_loading = false);

    static int get_num_reads(
            std::string data_path,
            std::optional<std::unordered_set<std::string>> read_list = std::nullopt,
            const std::unordered_set<std::string>& ignore_read_list = {},
            bool recursive_file_loading = false);

    static uint16_t get_sample_rate(std::string data_path, bool recursive_file_loading = false);

    std::string get_name() const { return "Dataloader"; }
    stats::NamedStats sample_stats() const;

private:
    void load_fast5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                               const std::vector<ReadID>& read_ids);
    void load_read_channels(std::string data_path, bool recursive_file_loading = false);
    Pipeline& m_pipeline;  // Where should the loaded reads go?
    std::atomic<size_t> m_loaded_read_count{0};
    std::string m_device;
    size_t m_num_worker_threads{1};
    size_t m_max_reads{0};
    std::optional<std::unordered_set<std::string>> m_allowed_read_ids;
    std::unordered_set<std::string> m_ignored_read_ids;

    std::unordered_map<std::string, channel_to_read_id_t> m_file_channel_read_order_map;
    int m_max_channel{0};
};

}  // namespace dorado
