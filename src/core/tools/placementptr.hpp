#pragma once

template <typename T>
class PlacementPtr
{
 public:
    PlacementPtr() = default;
    PlacementPtr(T* _ptr) : ptr(_ptr) {}
    PlacementPtr(const PlacementPtr&) = delete;
    PlacementPtr& operator=(const PlacementPtr&) = delete;

    PlacementPtr(PlacementPtr&& other) : ptr(other.ptr) { other.ptr = nullptr; }

    PlacementPtr& operator=(PlacementPtr&& other) {
        if (this == &other)
            return *this;
        close();
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }

    ~PlacementPtr() { close(); }

    operator T*() { return ptr; }
    operator bool() const { return ptr; }
    const T& operator*() const { return *ptr; }
    const T* operator->() const { return ptr; }
    T& operator*() { return *ptr; }
    T* operator->() { return ptr; }

 private:
    T* ptr = nullptr;

    void close() {
        if (ptr != nullptr) {
            ptr->~T();
            ptr = nullptr;
        }
    }
};
