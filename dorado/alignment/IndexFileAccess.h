#pragma once

#include "Minimap2IndexSupportTypes.h"
#include "Minimap2Options.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace dorado::alignment {

class Minimap2Index;

class IndexFileAccess {
    std::mutex m_mutex{};
    std::map<std::pair<std::string, Minimap2IndexOptions>, std::shared_ptr<Minimap2Index>>
            m_index_lut;

    std::shared_ptr<Minimap2Index> get_index_impl(const std::string& file,
                                                  const Minimap2IndexOptions& options);

public:
    IndexLoadResult load_index(const std::string& file,
                               const Minimap2Options& options,
                               int num_threads);

    // N.B. By contract load_index must be called prior to calling get_index.
    // i.e. will not return nullptr, there will be an assertion failure if load_index
    // has not been called for the file and index options.
    std::shared_ptr<const Minimap2Index> get_index(const std::string& file,
                                                   const Minimap2IndexOptions& options);
};

}  // namespace dorado::alignment
