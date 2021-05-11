#pragma once

#include <variant>
#include <vector>

template <typename T, size_t S>
class FixedArray
{
 public:
    explicit FixedArray(size_t _count) : count(_count) {
        if (count <= S)
            data = LocalData{};
        else
            data = DynamicData(count);
    }

    T& operator[](size_t i) {
        return count <= S ? std::get<LocalData>(data)[i] : std::get<DynamicData>(data)[i];
    }
    const T& operator[](size_t i) const {
        return count <= S ? std::get<LocalData>(data)[i] : std::get<DynamicData>(data)[i];
    }

    size_t size() const { return count; }
    T* data() { return &(*this)[0]; }
    const T* data() const { return &(*this)[0]; }

 private:
    struct LocalData
    {
        T values[S];
    };
    using DynamicData = std::vector<T>;
    size_t count;
    std::variant<LocalData, DynamicData> data;
};
