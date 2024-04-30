#pragma once

#include <map>
#include <memory>
#include <stdexcept>
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

    // returns the shared_ptr if registered otherwise returns nullptr
    template <typename T>
    std::shared_ptr<T> get_ptr() {
        auto registered_entry = m_contexts.find(typeid(T));
        if (registered_entry == m_contexts.end()) {
            return nullptr;
        }
        auto base = dynamic_cast<details::ContextHolder<T>*>(registered_entry->second.get());
        if (!base) {
            return nullptr;
        }
        return base->get_ptr();
    }

    // returns the shared_ptr if registered otherwise returns nullptr
    template <typename T>
    std::shared_ptr<T> get_ptr() const {
        auto registered_entry = m_contexts.find(typeid(T));
        if (registered_entry == m_contexts.end()) {
            return nullptr;
        }
        auto base = dynamic_cast<details::ContextHolder<T>*>(registered_entry->second.get());
        if (!base) {
            return nullptr;
        }
        return base->get_ptr();
    }

    // returns the value if registered otherwise throws std::out_of_range
    template <typename T>
    T& get() {
        auto registered_entry = m_contexts.find(typeid(T));
        if (registered_entry == m_contexts.end()) {
            throw std::out_of_range("Not a registered type");
        }
        auto base = dynamic_cast<details::ContextHolder<T>*>(registered_entry->second.get());
        if (!base) {
            throw std::out_of_range("Type not convertible");
        }
        return base->get();
    }

    // returns the value if registered otherwise throws std::out_of_range
    template <typename T>
    const T& get() const {
        auto registered_entry = m_contexts.find(typeid(T));
        if (registered_entry == m_contexts.end()) {
            throw std::out_of_range("Not a registered type");
        }
        auto base = dynamic_cast<details::ContextHolder<T>*>(registered_entry->second.get());
        if (!base) {
            throw std::out_of_range("Type not convertible");
        }
        return base->get();
    }

    template <typename T>
    bool exists() const {
        auto registered_entry = m_contexts.find(typeid(T));
        if (registered_entry == m_contexts.end()) {
            return false;
        }
        auto base = dynamic_cast<details::ContextHolder<T>*>(registered_entry->second.get());
        if (!base) {
            return false;
        }
        return true;
    }
};

}  // namespace dorado