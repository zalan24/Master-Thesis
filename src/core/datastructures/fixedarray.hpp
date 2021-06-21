#pragma once

#include <variant>
#include <vector>

template <typename T, size_t S>
class FixedArray
{
 public:
    explicit FixedArray(size_t _count) : count(_count) {
        if (count <= S)
            values = LocalData{};
        else
            values = DynamicData(count);
    }

    T& operator[](size_t i) {
        return count <= S ? std::get<LocalData>(values).values[i]
                          : std::get<DynamicData>(values)[i];
    }
    const T& operator[](size_t i) const {
        return count <= S ? std::get<LocalData>(values).values[i]
                          : std::get<DynamicData>(values)[i];
    }

    size_t size() const { return count; }
    T* data() { return &(*this)[0]; }
    const T* data() const { return &(*this)[0]; }

    bool empty() const { return size() == 0; }

 private:
    struct LocalData
    {
        T values[S];
    };
    using DynamicData = std::vector<T>;
    size_t count;
    std::variant<LocalData, DynamicData> values;
};
