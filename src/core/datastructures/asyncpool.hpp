#pragma once

#include <atomic>
#include <cassert>
#include <limits>
#include <shared_mutex>
#include <vector>

template <typename Child, typename ItemExt, bool MultiUse = false>
class AsyncPool
{
 public:
    AsyncPool() = default;

    AsyncPool(const AsyncPool&) = delete;
    AsyncPool& operator=(const AsyncPool&) = delete;

    using ItemIndex = uint32_t;
    static constexpr ItemIndex INVALID_ITEM = std::numeric_limits<ItemIndex>::max();

 protected:
    ~AsyncPool() { clearItems(); }

    void clearItems() {
        std::unique_lock<std::shared_mutex> lock(vectorMutex);
        assert(acquiredCount.load() == 0);
        items.clear();
        currentIndex = 0;
    }

    template <typename... Args>
    ItemIndex tryAcquireIndex(const Args&... args) noexcept {
        std::shared_lock<std::shared_mutex> lock(vectorMutex);
        size_t maxCount = items.size();
        for (size_t i = 0; i < maxCount; ++i) {
            ItemIndex index = currentIndex.fetch_add(1) % items.size();
            uint32_t expected = 0;
            if constexpr (MultiUse) {
                if (static_cast<Child*>(this)->canAcquire(items[index].itmExt, args...)) {
                    acquiredCount.fetch_add(1);
                    if (MultiUse)
                        items[index].useCount.fetch_add(1);
                    static_cast<Child*>(this)->acquireExt(items[index].itmExt, args...);
                    return index;
                }
            }
            else {
                if (items[index].useCount.compare_exchange_weak(expected, 1)) {
                    if (static_cast<Child*>(this)->canAcquire(items[index].itmExt, args...)) {
                        acquiredCount.fetch_add(1);
                        static_cast<Child*>(this)->acquireExt(items[index].itmExt, args...);
                        return index;
                    }
                    else
                        items[index].useCount = 0;
                }
            }
        }
        return INVALID_ITEM;
    }

    template <typename... Args>
    ItemIndex acquireIndex(const Args&... args) {
        ItemIndex ret = tryAcquireIndex(args...);
        if (ret != INVALID_ITEM)
            return ret;
        std::unique_lock<std::shared_mutex> lock(vectorMutex);
        items.emplace_back();
        items.back().useCount.fetch_add(1);
        acquiredCount.fetch_add(1);
        ret = ItemIndex(items.size() - 1);
        if (!static_cast<Child*>(this)->canAcquire(items[ret].itmExt, args...))
            throw std::runtime_error("Newly created item is not suitable to be acquired");
        static_cast<Child*>(this)->acquireExt(items[ret].itmExt, args...);
        return ItemIndex(ret);
    }

    ItemExt& getItem(ItemIndex index) { return items[index].itmExt; }
    const ItemExt& getItem(ItemIndex index) const { return items[index].itmExt; }

    void release(ItemIndex itemIndex) {
        std::shared_lock<std::shared_mutex> lock(vectorMutex);
        static_cast<Child*>(this)->releaseExt(items[itemIndex].itmExt);
        assert(items[itemIndex].useCount.fetch_sub(1) == 1 || MultiUse);
        assert(acquiredCount.fetch_sub(1) > 0);
    }

    mutable std::shared_mutex vectorMutex;

 private:
    struct ItemImpl
    {
        std::atomic<uint32_t> useCount = {false};
        ItemExt itmExt;
        ItemImpl() = default;
        ItemImpl(const ItemImpl&) = delete;
        ItemImpl& operator=(const ItemImpl&) = delete;
        ItemImpl(ItemImpl&& other) noexcept
          : useCount(other.useCount.load()), itmExt(std::move(other.itmExt)) {}
        ItemImpl& operator=(ItemImpl&& other) noexcept {
            if (this == &other)
                return *this;
            useCount = other.useCount.load();
            itmExt = std::move(other.itmExt);
            return *this;
        }
    };
    std::atomic<ItemIndex> currentIndex = 0;
    std::atomic<ItemIndex> acquiredCount = 0;
    std::vector<ItemImpl> items;
};
