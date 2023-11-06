#pragma once

#include "Minimap2Index.h"
#include "Minimap2Options.h"
#include "ReadPipeline.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace dorado::alignment {

class IndexFileAccess {
    std::mutex m_mutex{};
    std::map<std::pair<std::string, Minimap2IndexOptions>, std::shared_ptr<Minimap2Index>>
            m_index_lut{};

    std::shared_ptr<Minimap2Index> get_index_impl(const std::string& file,
                                                  const Minimap2IndexOptions& options);

public:
    IndexLoadResult load_index(const std::string& file,
                               const Minimap2Options& options,
                               int num_threads);

    std::shared_ptr<Minimap2Index> get_index(const std::string& file,
                                             const Minimap2IndexOptions& options);
};

}  // namespace dorado::alignment
