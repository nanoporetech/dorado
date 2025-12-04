#pragma once

#include "utils/stats.h"

#include <functional>
#include <variant>

namespace dorado {

class HtsData;

namespace hts_writer {

using Processable = std::variant<std::reference_wrapper<HtsData>>;

template <typename Visitor>
void dispatch_processable(const Processable& item, Visitor&& visitor) {
    // Visit and dispatch to type-specific handlers.
    std::visit(
            [&](auto ref_wrapper) {
                using T = std::decay_t<decltype(ref_wrapper.get())>;
                T& data = ref_wrapper.get();
                visitor(data);
            },
            item);
}

class IWriter {
public:
    virtual ~IWriter() = default;
    virtual void process(const Processable item) = 0;
    virtual void shutdown() = 0;

    virtual std::string get_name() const = 0;
    virtual stats::NamedStats sample_stats() const = 0;
};

class NullWriter : public IWriter {
public:
    void process([[maybe_unused]] const Processable item) override {};
    void shutdown() override {};

    std::string get_name() const override { return "NullWriter"; }
    stats::NamedStats sample_stats() const override { return {}; }
};

}  // namespace hts_writer
}  // namespace dorado
