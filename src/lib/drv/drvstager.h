#pragma once

#include <exclusive.h>

#include "drv_wrappers.h"
#include "drverror.h"

namespace drv
{
// Buffer will be exclusive
class Stager : private Exclusive
{
 public:
    struct Range
    {
        DeviceSize offset;
        DeviceSize size;
    };

    explicit Stager(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
                    QueueFamilyPtr family, QueuePtr queue, DeviceSize size);

    ~Stager();

    Stager(const Stager&) = delete;
    Stager& operator=(const Stager&) = delete;

    Stager(Stager&&) = default;
    Stager& operator=(Stager&&) = default;

    MemoryMapper mapMemory();
    MemoryMapper mapMemory(const Range& range);

    void write(DeviceSize size, const void* data);
    void write(const Range& range, const void* data);

    void push(BufferPtr buffer);
    void pushTo(BufferPtr buffer, const Range& to);
    void pushFrom(BufferPtr buffer, const Range& from);
    void push(BufferPtr buffer, const Range& from, const Range& to);

    void read(DeviceSize size, void* data) const;
    void read(const Range& range, void* data) const;

    void pull(BufferPtr buffer);
    void pullTo(BufferPtr buffer, const Range& to);
    void pullFrom(BufferPtr buffer, const Range& from);
    void pull(BufferPtr buffer, const Range& from, const Range& to);

    drv::FenceWaitResult wait(unsigned long long int timeOut = 0) const;

    template <typename T>
    void upload(BufferPtr buffer, unsigned int count, const T* data) {
        upload(buffer, count, data, 0);
    }

    template <typename T>
    void upload(BufferPtr buffer, unsigned int count, const T* data, DeviceSize offset) {
        CHECK_THREAD;
        const DeviceSize s = sizeof(T) * count;
        drv_assert(s <= size, "Staging buffer is too small");
        write(s, data);
        pushTo(buffer, Range{offset, s});
    }

    template <typename T>
    void download(BufferPtr buffer, unsigned int count, T* data) {
        download(buffer, count, data, 0);
    }

    template <typename T>
    void download(BufferPtr buffer, unsigned int count, T* data, DeviceSize offset) {
        CHECK_THREAD;
        const DeviceSize s = sizeof(T) * count;
        drv_assert(s <= size, "Staging buffer is too small");
        wait();
        pullFrom(buffer, Range{offset, s});
        read(s, data);
    }

    void download(BufferPtr buffer, DeviceSize _size);
    void download(BufferPtr buffer, DeviceSize _size, DeviceSize offset);

 private:
    DeviceSize size;
    LogicalDevicePtr device;
    QueuePtr queue;
    BufferSet bufferSet;
    BufferPtr stagingBuffer;
    Fence fence;
    CommandPool commandPool;
    CommandBuffer commandBuffer;

    mutable bool fenceWaited = true;

    MemoryMapper mapMemoryImpl() const;
    MemoryMapper mapMemoryImpl(const Range& range) const;
};
}  // namespace drv
