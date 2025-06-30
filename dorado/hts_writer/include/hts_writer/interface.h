#pragma once

#include "hts_utils/hts_types.h"

namespace dorado {
namespace hts_writer {

using Processable = std::variant<std::reference_wrapper<const HtsData>>;

template <typename Visitor>
void dispatch_processable(const Processable& item, Visitor&& visitor) {
    // Visit and dispatch to type-specific handlers.
    std::visit(
            [&](auto ref_wrapper) {
                using T = std::decay_t<decltype(ref_wrapper.get())>;
                const T& data = ref_wrapper.get();
                visitor(data);
            },
            item);
}

class IWriter {
public:
    virtual ~IWriter() = default;
    virtual void init() = 0;
    virtual void process(const Processable item) = 0;
    virtual void shutdown() = 0;
};

class NullWriter : public IWriter {
public:
    void init() {};
    void process(const Processable _) {};
    void shutdown() {};
};

}  // namespace hts_writer
}  // namespace dorado
