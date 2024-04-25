#pragma once

#include <map>
#include <memory>
#include <typeindex>

namespace dorado {

namespace details {

class ContextBase {
public:
    virtual ~ContextBase() = default;
};

template <typename T>
class ContextHolder : public ContextBase {
    std::shared_ptr<T> m_context;

public:
    ContextHolder(std::shared_ptr<T> context) : m_context(std::move(context)) {}

    std::shared_ptr<T> get_ptr() { return m_context; }

    const T& get() const { return *m_context; }

    T& get() { return *m_context; }
};

}  // namespace details

class ContextContainer {
    std::map<std::type_index, std::unique_ptr<details::ContextBase>> m_contexts{};

public:
    /// N.B. will replace any existing concrete context already registered.
    /// If this is not the desired behaviour check exists() before calling.
    template <typename ALIAS, typename IMPL>
    void register_context(std::shared_ptr<IMPL> context) {
        auto context_as_alias_type = std::static_pointer_cast<ALIAS>(context);
        m_contexts[typeid(ALIAS)] =
                std::make_unique<details::ContextHolder<ALIAS>>(std::move(context_as_alias_type));
    }

    template <typename T>
    std::shared_ptr<T> get_ptr() {
        auto& base = m_contexts.at(typeid(T));
        return dynamic_cast<details::ContextHolder<T>*>(base.get())->get_ptr();
    }

    template <typename T>
    std::shared_ptr<T> get_ptr() const {
        auto& base = m_contexts.at(typeid(T));
        return dynamic_cast<details::ContextHolder<T>*>(base.get())->get_ptr();
    }

    template <typename T>
    T& get() {
        auto& base = m_contexts.at(typeid(T));
        return dynamic_cast<details::ContextHolder<T>*>(base.get())->get();
    }

    template <typename T>
    const T& get() const {
        auto& base = m_contexts.at(typeid(T));
        return dynamic_cast<details::ContextHolder<T>*>(base.get())->get();
    }

    template <typename T>
    bool exists() const {
        return m_contexts.count(typeid(T)) > 0;
    }
};

}  // namespace dorado