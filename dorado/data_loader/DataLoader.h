#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dorado {

class MessageSink;
struct ReadGroup;

class DataLoader {
public:
    DataLoader(MessageSink& read_sink,
               const std::string& device,
               size_t num_worker_threads,
               size_t max_reads = 0,
               std::unordered_set<std::string> read_list = std::unordered_set<std::string>());
    void load_reads(const std::string& path, bool recursive_file_loading = false);

    static std::unordered_map<std::string, ReadGroup> load_read_groups(
            std::string data_path,
            std::string model_path,
            bool recursive_file_loading = false);

    static int get_num_reads(
            std::string data_path,
            std::unordered_set<std::string> read_list = std::unordered_set<std::string>(),
            bool recursive_file_loading = false);

private:
    void load_fast5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file(const std::string& path);
    MessageSink& m_read_sink;  // Where should the loaded reads go?
    size_t m_loaded_read_count{0};
    std::string m_device;
    size_t m_num_worker_threads{1};
    size_t m_max_reads{0};
    std::unordered_set<std::string> m_allowed_read_ids;
};

}  // namespace dorado
