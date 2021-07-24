#pragma once

#include <drv_wrappers.h>

#include <garbagesystem.h>

namespace res
{
template <typename T>
class GarbageResource
{
 public:
    using value_type = T;

    GarbageResource() = default;
    GarbageResource(GarbageSystem* _garbageSystem, T&& _data)
      : garbageSystem(_garbageSystem), data(std::move(data)) {}
    template <typename... Args>
    explicit GarbageResource(GarbageSystem* _garbageSystem, Args&&... args)
      : garbageSystem(_garbageSystem), data(std::forward<Args>(args)...) {}
    GarbageResource(GarbageResource&& other)
      : garbageSystem(other.garbageSystem), data(std::move(other.data)) {
        other.garbageSystem = nullptr;
    }
    GarbageResource& operator=(GarbageResource&& other) {
        if (this == &other)
            return *this;
        close();
        garbageSystem = other.garbageSystem;
        data = std::move(other.data);
        other.garbageSystem = nullptr;
        return *this;
    }
    ~GarbageResource() { close(); }
    void close() {
        if (garbageSystem != nullptr) {
            garbageSystem->useGarbage([this](Garbage* trash) { trash->release(std::move(data)); });
            garbageSystem = nullptr;
        }
    }

    operator T&() { return data; }
    operator const T&() const { return data; }

    T& get() { return data; }
    const T& get() const { return data; }

    operator bool() const { return garbageSystem != nullptr; }

 private:
    GarbageSystem* garbageSystem = nullptr;
    T data;
};

using ImageSet = GarbageResource<drv::ImageSet>;
using BufferSet = GarbageResource<drv::BufferSet>;
using ImageView = GarbageResource<drv::ImageView>;
using Framebuffer = GarbageResource<drv::Framebuffer>;

}  // namespace res
