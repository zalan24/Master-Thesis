#pragma once

#include "util.hpp"

template <typename T>
class Singleton
{
 public:
    static T* getSingleton() {
        ASSERT(instance != nullptr);
        return instance;
    }
    Singleton() {
        ASSERT(instance == nullptr);
        instance = static_cast<T*>(this);
    }
    ~Singleton() {
        if (instance == static_cast<T*>(this))
            instance = nullptr;
    }
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    Singleton(Singleton&& other) {
        if (instance == static_cast<T*>(&other))
            instance = static_cast<T*>(this);
    }
    Singleton& operator=(Singleton&& other) {
        // This is compatible with self-assignment
        if (instance == static_cast<T*>(&other))
            instance = static_cast<T*>(this);
    }

 private:
    static T* instance;
};
