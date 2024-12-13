#pragma once

#include "file_info/file_info.h"
#include "models/kits.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
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

    // Holds the directory entries for the pod5 or fast5 files from the input path.
    // If there are both fast5 and pod5 only the pod5 will be held.
    // Used as input to loading reads to ensure any mixed input files types has been
    // correctly filtered to either pod5 or fast5 only.
    class InputFiles final {
        std::vector<std::filesystem::directory_entry> m_entries;

    public:
        static std::optional<InputFiles> search(const std::filesystem::path& path, bool recursive);
        const std::vector<std::filesystem::directory_entry>& get() const;
    };

    void load_reads(const InputFiles& input_files, ReadOrder traversal_order);

    std::string get_name() const { return "Dataloader"; }
    stats::NamedStats sample_stats() const;

    struct ReadSortInfo {
        std::string read_id;
        int32_t mux;
        uint32_t read_number;
    };

    using ReadInitialiserF = std::function<void(ReadCommon&)>;
    void add_read_initialiser(ReadInitialiserF func) {
        m_read_initialisers.push_back(std::move(func));
    }

    // Retrieves the pod5 or fast5 entries from the input path.
    // If there are both fast5 and pod5 only the pod5 will be returned.
    static std::vector<std::filesystem::directory_entry> get_directory_entries(
            const std::filesystem::path& path,
            bool recursive_file_loading);

private:
    void load_fast5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                               const std::vector<ReadID>& read_ids);
    void load_read_channels(const std::vector<std::filesystem::directory_entry>& files);

    void load_reads_by_channel(const std::vector<std::filesystem::directory_entry>& files);
    void load_reads_unrestricted(const std::vector<std::filesystem::directory_entry>& files);

    void initialise_read(ReadCommon& read) const;

    Pipeline& m_pipeline;  // Where should the loaded reads go?
    std::atomic<size_t> m_loaded_read_count{0};
    std::string m_device;
    size_t m_num_worker_threads{1};
    size_t m_max_reads{0};
    std::optional<std::unordered_set<std::string>> m_allowed_read_ids;
    std::unordered_set<std::string> m_ignored_read_ids;

    std::unordered_map<std::string, channel_to_read_id_t> m_file_channel_read_order_map;
    std::unordered_map<int, std::vector<ReadSortInfo>> m_reads_by_channel;
    std::unordered_map<std::string, size_t> m_read_id_to_index;
    int m_max_channel{0};

    std::vector<ReadInitialiserF> m_read_initialisers;

    // Issue warnings if read is potentially problematic
    void check_read(const SimplexReadPtr& read);
    // A flag to warn only once if the data chemsitry is known
    std::atomic<bool> m_log_unknown_chemistry{true};
};

}  // namespace dorado
