#pragma once

#include "utils/stats.h"
#include "utils/types.h"

#include <cxxpool.h>

#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado {

class Pipeline;
class ReadCommon;
class SimplexRead;
using SimplexReadPtr = std::unique_ptr<SimplexRead>;

constexpr size_t POD5_READ_ID_SIZE = 16;
using ReadID = std::array<uint8_t, POD5_READ_ID_SIZE>;
typedef std::map<int, std::vector<ReadID>> channel_to_read_id_t;

class DataLoader {
public:
    DataLoader(Pipeline& pipeline,
               const std::string& device,
               size_t num_worker_threads,
               size_t max_reads,
               std::optional<std::unordered_set<std::string>> read_list,
               std::unordered_set<std::string> read_ignore_list);
    ~DataLoader() = default;

    // Holds the directory entries for the pod5 files from the input path.
    class InputFiles final {
        std::vector<std::filesystem::directory_entry> m_entries;

    public:
        static InputFiles search_pod5s(const std::filesystem::path& path, bool recursive);
        const std::vector<std::filesystem::directory_entry>& get() const;
    };

    void load_reads(const InputFiles& input_files, ReadOrder traversal_order);

    struct ReadSortInfo {
        std::string read_id;
        int32_t mux;
        uint32_t read_number;
    };

    using ReadInitialiserF = std::function<void(ReadCommon&)>;
    void add_read_initialiser(ReadInitialiserF func) {
        m_read_initialisers.push_back(std::move(func));
    }

private:
    void load_pod5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                               const std::vector<ReadID>& read_ids);
    void load_read_channels(const std::vector<std::filesystem::directory_entry>& files);

    void load_reads_by_channel(const std::vector<std::filesystem::directory_entry>& files);
    void load_reads_unrestricted(const std::vector<std::filesystem::directory_entry>& files);

    void initialise_read(ReadCommon& read) const;

    Pipeline& m_pipeline;  // Where should the loaded reads go?
    size_t m_loaded_read_count{0};
    std::string m_device;
    cxxpool::thread_pool m_thread_pool;
    size_t m_max_reads{0};
    std::optional<std::unordered_set<std::string>> m_allowed_read_ids;
    std::unordered_set<std::string> m_ignored_read_ids;

    std::unordered_map<std::string, channel_to_read_id_t> m_file_channel_read_order_map;
    std::unordered_map<int, std::vector<ReadSortInfo>> m_reads_by_channel;
    std::unordered_map<std::string, size_t> m_read_id_to_index;
    int m_max_channel{0};

    std::vector<ReadInitialiserF> m_read_initialisers;
    std::atomic<bool> m_stop_loading{false};
    // Issue warnings if read is potentially problematic
    void check_read(const SimplexReadPtr& read);
    // A flag to warn only once if the data chemsitry is known
    std::atomic<bool> m_log_unknown_chemistry{true};
};

}  // namespace dorado
