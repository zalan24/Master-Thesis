#pragma once

#include <atomic>
#include <cassert>
#include <limits>
#include <shared_mutex>
#include <vector>

#include <logger.h>

template <typename Child, typename ItemExt>
class AsyncPool
{
 public:
    explicit AsyncPool(const char* _name, uint32_t _warnLimit = 0)
      : name(_name), warnLimit(_warnLimit) {}

    AsyncPool(const AsyncPool&) = delete;
    AsyncPool& operator=(const AsyncPool&) = delete;

    using ItemIndex = uint32_t;
    static constexpr ItemIndex INVALID_ITEM = std::numeric_limits<ItemIndex>::max();

    void release(ItemIndex itemIndex) {
        std::shared_lock<std::shared_mutex> lock(vectorMutex);
        static_cast<Child*>(this)->releaseExt(items[itemIndex].itmExt);
        assert(items[itemIndex].used.load() == true);
        items[itemIndex].used = false;
        ItemIndex count = acquiredCount.fetch_sub(1);
        assert(count > 0);
    }

 protected:
    ~AsyncPool() { clearItems(); }

    void clearItems() {
        std::unique_lock<std::shared_mutex> lock(vectorMutex);
        ItemIndex count = acquiredCount.load();
        if (count > 0)
            LOG_F(ERROR, "Not all items were freed. Remaining: %d <%s>", count, name);
        assert(count == 0);
        items.clear();
        currentIndex = 0;
    }

    template <typename... Args>
    ItemIndex tryAcquireIndex(const Args&... args) noexcept {
        std::shared_lock<std::shared_mutex> lock(vectorMutex);
        size_t maxCount = items.size();
        for (size_t i = 0; i < maxCount; ++i) {
            ItemIndex index = currentIndex.fetch_add(1) % items.size();
            bool expected = false;
            if (items[index].used.compare_exchange_weak(expected, true)) {
                if (static_cast<Child*>(this)->canAcquire(items[index].itmExt, args...)) {
                    acquiredCount.fetch_add(1);
                    static_cast<Child*>(this)->acquireExt(items[index].itmExt, args...);
                    return index;
                }
                else
                    items[index].used = false;
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
        ret = ItemIndex(items.size() - 1);
        if (!static_cast<Child*>(this)->canAcquire(items[ret].itmExt, args...))
            throw std::runtime_error("Newly created item is not suitable to be acquired");
        static_cast<Child*>(this)->acquireExt(items[ret].itmExt, args...);
        if (items.size() == warnLimit)
            LOG_F(WARNING, "There are too many items in this pool: %lld <%s>", items.size(), name);
        items.back().used = true;
        acquiredCount.fetch_add(1);
        return ItemIndex(ret);
    }

    ItemExt& getItem(ItemIndex index) { return items[index].itmExt; }
    const ItemExt& getItem(ItemIndex index) const { return items[index].itmExt; }

    mutable std::shared_mutex vectorMutex;

 private:
    struct ItemImpl
    {
        std::atomic<bool> used = {false};
        ItemExt itmExt;
        ItemImpl() = default;
        ItemImpl(const ItemImpl&) = delete;
        ItemImpl& operator=(const ItemImpl&) = delete;
        ItemImpl(ItemImpl&& other) noexcept
          : used(other.used.load()), itmExt(std::move(other.itmExt)) {}
        ItemImpl& operator=(ItemImpl&& other) noexcept {
            if (this == &other)
                return *this;
            used = other.used.load();
            itmExt = std::move(other.itmExt);
            return *this;
        }
    };
    const char* name;
    std::atomic<ItemIndex> currentIndex = 0;
    std::atomic<ItemIndex> acquiredCount = 0;
    std::vector<ItemImpl> items;
    uint32_t warnLimit = 0;
};
