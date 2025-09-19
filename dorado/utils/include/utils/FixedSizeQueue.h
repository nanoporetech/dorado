#pragma once

#include <cassert>
#include <memory>
#include <new>
#include <type_traits>

namespace dorado::utils {

template <typename Item>
class FixedSizeQueue {
    // Fake item buffer so that items don't have to be default constructible.
    struct ItemBuffer {
        alignas(Item) std::byte data[sizeof(Item)];
    };

    // We allocate 1 extra element so that we can easily distinguish between empty and full.
    // read==write is empty, read==write+1 is full.
    // Typically we have large queues of small objects, so an additional element isn't a big deal.
    const std::size_t m_capacity;
    const std::unique_ptr<ItemBuffer[]> m_items;
    std::size_t m_read_idx = 0;
    std::size_t m_write_idx = 0;

    static_assert(std::is_move_constructible_v<Item>);

public:
    explicit FixedSizeQueue(std::size_t capacity)
            // See above for the reasoning behind this +1.
            : m_capacity(capacity + 1), m_items(std::make_unique<ItemBuffer[]>(m_capacity)) {}

    ~FixedSizeQueue() { clear(); }

    // Push an item onto the queue.
    // Requires that the queue isn't full.
    void push(Item item) {
        assert(!full());

        // Construct the new item in-place.
        auto *buffer = std::addressof(m_items[m_write_idx].data);
        new (buffer) Item(std::move(item));

        // Move the write pointer.
        m_write_idx = (m_write_idx + 1) % m_capacity;
    }

    // Pop an item from the queue.
    // Requires that the queue isn't empty.
    [[nodiscard]] Item pop() {
        assert(!empty());

        // Pop the item.
        auto *buffer = std::addressof(m_items[m_read_idx].data);
        auto *item_ptr = std::launder(reinterpret_cast<Item *>(buffer));
        Item item(std::move(*item_ptr));

        // Destroy it so that the slot is free to be reused.
        item_ptr->~Item();

        // Move the read pointer.
        m_read_idx = (m_read_idx + 1) % m_capacity;
        return item;
    }

    // Clear the queue.
    void clear() {
        while (!empty()) {
            (void)pop();
        }
    }

    // See if the queue is empty.
    [[nodiscard]] bool empty() const { return m_read_idx == m_write_idx; }

    // See if the queue is full.
    [[nodiscard]] bool full() const {
        const std::size_t next_write = (m_write_idx + 1) % m_capacity;
        return next_write == m_read_idx;
    }

    // Determine the current size of the queue.
    [[nodiscard]] std::size_t size() const {
        if (m_write_idx >= m_read_idx) {
            return m_write_idx - m_read_idx;
        } else {
            return m_capacity - (m_read_idx - m_write_idx);
        }
    }

    // How many items this queue can hold.
    [[nodiscard]] std::size_t capacity() const { return m_capacity - 1; }
};

}  // namespace dorado::utils
