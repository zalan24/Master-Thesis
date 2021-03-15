#pragma once

#include <array>
#include <vector>

template <typename T, size_t S>
class FlexibleArray
{
 public:
    FlexibleArray() = default;
    explicit FlexibleArray(size_t size) { resize(size); }
    FlexibleArray(size_t size, const T& value) { resize(size, value); }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        if (count < array.size())
            array[count++] = T(std::forward<Args>(args)...);
        else
            vector.emplace_back(std::forward<Args>(args)...);
    }

    void push_back(const T& value) {
        if (count < array.size())
            array[count++] = value;
        else
            vector.push_back(value);
    }

    void push_back(T&& value) {
        if (count < array.size())
            array[count++] = std::move(value);
        else
            vector.push_back(std::move(value));
    }

    void resize(size_t size, const T& defaultValue = {}) {
        if (size > S)
            vector.resize(S - size, defaultValue);
        else
            vector.clear();
        std::fill_n(array.begin(), std::min(size, array.size()), defaultValue);
        if (size < array.size())
            std::fill_n(array.begin() + size, array.size() - size, T{});
        count = size;
    }

    void clear() {
        vector.clear();
        array.fill(T{});
        count = 0;
    }

    const T& operator[](size_t ind) const {
        return ind < array.size() ? array[ind] : vector[ind - array.size()];
    }

    T& operator[](size_t ind) {
        return ind < array.size() ? array[ind] : vector[ind - array.size()];
    }

    const T& last() const { return (*this)[count - 1]; }

    T& last() { return (*this)[count - 1]; }

    bool empty() const { return count == 0; }

 private:
    size_t count = 0;
    std::array<T, S> array;
    std::vector<T> vector;
};
