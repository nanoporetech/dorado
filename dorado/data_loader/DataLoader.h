#pragma once

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

struct Pod5FileReader;

namespace dorado {

class Pipeline;
class ReadCommon;
class SimplexRead;
using SimplexReadPtr = std::unique_ptr<SimplexRead>;

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
               size_t max_reads,
               std::optional<std::unordered_set<std::string>> read_list,
               std::unordered_set<std::string> read_ignore_list);
    ~DataLoader() = default;
    void load_reads(const std::filesystem::path& path,
                    bool recursive_file_loading,
                    ReadOrder traversal_order);

    static std::unordered_map<std::string, ReadGroup> load_read_groups(
            const std::filesystem::path& data_path,
            std::string model_name,
            std::string modbase_model_names,
            bool recursive_file_loading);

    static int get_num_reads(const std::filesystem::path& data_path,
                             std::optional<std::unordered_set<std::string>> read_list,
                             const std::unordered_set<std::string>& ignore_read_list,
                             bool recursive_file_loading);

    static bool is_read_data_present(const std::filesystem::path& data_path,
                                     bool recursive_file_loading);

    static uint16_t get_sample_rate(const std::filesystem::path& data_path,
                                    bool recursive_file_loading);

    // Inspects the sequencing data metadata to determine the sequencing chemistry used.
    // Calls get_sequencing_chemistries but will error if the data is inhomogeneous
    static models::Chemistry get_unique_sequencing_chemisty(const std::string& data,
                                                            bool recursive_file_loading);

    static std::set<models::ChemistryKey> get_sequencing_chemistries(
            const std::filesystem::path& data_path,
            bool recursive_file_loading);

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

private:
    void load_fast5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                               const std::vector<ReadID>& read_ids);
    void load_read_channels(const std::filesystem::path& data_path, bool recursive_file_loading);

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
