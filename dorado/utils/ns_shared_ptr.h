#pragma once

#include <utility>

namespace dorado::utils {

// Backport of NS::SharedPtr<> that's introduced in a more recent version of metal-cpp.
// Semantics are the same as a std::shared_ptr<>.
template <typename T>
class SharedPtr {
    T *m_object{nullptr};

    template <typename U>
    friend SharedPtr<U> TransferPtr(U *object);

    template <typename U>
    friend SharedPtr<U> RetainPtr(U *object);

    void retain() {
        if (m_object) {
            m_object->retain();
        }
    }
    void release() {
        if (m_object) {
            m_object->release();
        }
    }

public:
    SharedPtr() = default;
    ~SharedPtr() { release(); }

    SharedPtr(SharedPtr &other) : SharedPtr() { operator=(other); }
    SharedPtr(SharedPtr &&other) : SharedPtr() { operator=(std::move(other)); }

    SharedPtr &operator=(SharedPtr &other) {
        other.retain();
        release();
        m_object = other.m_object;
        return *this;
    }
    SharedPtr &operator=(SharedPtr &&other) {
        release();
        m_object = other.m_object;
        other.m_object = nullptr;
        return *this;
    }

    explicit operator bool() const { return m_object != nullptr; }
    T *get() const { return m_object; }
    T *operator->() const { return m_object; }
};

// Transfer ownership of an object to a SharedPtr (ie without adjusting its reference count).
template <typename T>
SharedPtr<T> TransferPtr(T *object) {
    SharedPtr<T> ptr;
    ptr.m_object = object;
    return ptr;
}

// Take shared ownership of an existing object.
template <typename T>
SharedPtr<T> RetainPtr(T *object) {
    SharedPtr<T> ptr;
    object->retain();
    ptr.m_object = object;
    return ptr;
}

}  // namespace dorado::utils
