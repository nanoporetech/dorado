#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeindex>

namespace dorado {

namespace details {

class ContextBase {
public:
    virtual ~ContextBase() = default;
    virtual bool is_const() const = 0;
};

template <typename T>
class ContextHolder final : public ContextBase {
    std::shared_ptr<T> m_context;

public:
    ContextHolder(std::shared_ptr<T> context) : m_context(std::move(context)) {}

    std::shared_ptr<T> get_ptr() { return m_context; }

    const T& get() const { return *m_context; }

    T& get() { return *m_context; }

    bool is_const() const override { return std::is_const_v<T>; }
};

}  // namespace details

class ContextContainer final {
    std::map<std::type_index, std::unique_ptr<details::ContextBase>> m_contexts{};

    template <typename T>
    details::ContextBase* get_base() const {
        auto registered_entry = m_contexts.find(typeid(T));
        if (registered_entry == m_contexts.end()) {
            return nullptr;
        }

        if constexpr (!std::is_const_v<T>) {
            if (registered_entry->second->is_const()) {
                return nullptr;
            }
        }

        return registered_entry->second.get();
    }

public:
    /// N.B. will replace any existing concrete context already registered.
    /// If this is not the desired behaviour check exists() before calling.
    template <typename ALIAS, typename IMPL>
    void register_context(const std::shared_ptr<IMPL>& context) {
        auto context_as_alias_type = std::static_pointer_cast<ALIAS>(context);
        m_contexts[typeid(ALIAS)] =
                std::make_unique<details::ContextHolder<ALIAS>>(std::move(context_as_alias_type));
    }

    // returns the shared_ptr if registered otherwise returns nullptr
    template <typename T>
    std::shared_ptr<T> get_ptr() const {
        auto base_entry = get_base<T>();
        if (!base_entry) {
            return nullptr;
        }

        if (std::is_const_v<T> == base_entry->is_const()) {
            auto holder = dynamic_cast<details::ContextHolder<T>*>(base_entry);
            return holder ? holder->get_ptr() : nullptr;
        }

        auto holder = dynamic_cast<details::ContextHolder<std::remove_const_t<T>>*>(base_entry);
        return holder ? holder->get_ptr() : nullptr;
    }

    // returns the value if registered otherwise throws std::out_of_range
    template <typename T>
    T& get() const {
        auto base_entry = get_base<T>();
        if (!base_entry) {
            throw std::out_of_range("Not a registered type");
        }

        if (std::is_const_v<T> == base_entry->is_const()) {
            auto holder = dynamic_cast<details::ContextHolder<T>*>(base_entry);
            if (!holder) {
                throw std::out_of_range("Type not convertible");
            }
            return holder->get();
        }

        auto holder = dynamic_cast<details::ContextHolder<std::remove_const_t<T>>*>(base_entry);
        if (!holder) {
            throw std::out_of_range("Type not convertible");
        }
        return holder->get();
    }

    template <typename T>
    bool exists() const {
        auto base_entry = get_base<T>();
        if (!base_entry) {
            return false;
        }

        if (std::is_const_v<T> == base_entry->is_const()) {
            auto holder = dynamic_cast<details::ContextHolder<T>*>(base_entry);
            return holder != nullptr;
        }

        auto holder = dynamic_cast<details::ContextHolder<std::remove_const_t<T>>*>(base_entry);
        return holder != nullptr;
    }
};

}  // namespace dorado