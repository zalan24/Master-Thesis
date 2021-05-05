#pragma once

#include <algorithm>
#include <vector>

template <typename T, size_t S>
class FlexibleArray
{
 public:
    FlexibleArray() {}
    explicit FlexibleArray(size_t size) { resize(size); }
    FlexibleArray(size_t size, const T& value) { resize(size, value); }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        if (count < S)
            array[count] = T(std::forward<Args>(args)...);
        else
            vector.emplace_back(std::forward<Args>(args)...);
        count++;
    }

    void push_back(const T& value) {
        if (count < S)
            array[count] = value;
        else
            vector.push_back(value);
        count++;
    }

    void push_back(T&& value) {
        if (count < S)
            array[count] = std::move(value);
        else
            vector.push_back(std::move(value));
        count++;
    }

    void reserve(size_t size) {
        if (size > S)
            vector.reserve(size - S);
    }

    template <typename... Args>
    void resize(size_t size, Args&&... args) {
        if (size > S)
            vector.resize(S - size);
        else
            vector.clear();
        for (size_t i = size; i < count && i < S; ++i)
            array[i] = T(std::forward<Args>(args)...);
        count = size;
    }

    template <typename... Args>
    void clear(Args&&... args) {
        vector.clear();
        for (size_t i = 0; i < count && i < S; ++i)
            array[i] = T(std::forward<Args>(args)...);
        count = 0;
    }

    const T& operator[](size_t ind) const { return ind < S ? array[ind] : vector[ind - S]; }

    T& operator[](size_t ind) { return ind < S ? array[ind] : vector[ind - S]; }

    const T& back() const { return (*this)[count - 1]; }
    T& back() { return (*this)[count - 1]; }
    const T& front() const { return (*this)[0]; }
    T& front() { return (*this)[0]; }

    template <typename... Args>
    void pop_back(Args&&... args) {
        if (!empty()) {
            if (count > S)
                vector.pop_back();
            else
                back() = T(std::forward<Args>(args)...);
            count--;
        }
    }

    bool empty() const { return count == 0; }

    size_t size() const { return count; }
    size_t fixedSize() const {return S;}

 private:
    size_t count = 0;
    T array[S];
    std::vector<T> vector;
};
