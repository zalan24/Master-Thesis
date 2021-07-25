#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

#include <flexiblearray.hpp>

#include <drvimage_types.h>
#include <drvresourceptrs.hpp>

namespace drv
{
class ResourceLocker;

class ResourceLockerDescriptor
{
 public:
    enum UsageMode
    {
        NONE = 0,
        READ = 1,
        WRITE = 2,
        READ_WRITE = 3
    };

    void addImage(drv::ImagePtr image, uint32_t layerCount, uint32_t layer, uint32_t mip,
                  drv::AspectFlagBits aspect, UsageMode usage) noexcept;
    void addImage(drv::ImagePtr image, const drv::ImageSubresourceSet& subresources,
                  UsageMode usage) noexcept;

    UsageMode getImageUsage(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                            drv::AspectFlagBits aspect) const;
    UsageMode getImageUsage(uint32_t index, uint32_t layer, uint32_t mip,
                            drv::AspectFlagBits aspect) const;

    enum ConflictType
    {
        NONE_NO_OVERLAP,
        NONE_COMMON_READ,
        CONFLICT_WRITE
    };

    ConflictType findConflict(const ResourceLockerDescriptor* other) const;

    bool empty() const;
    ImagePtr getImage(uint32_t index) const;
    BufferPtr getBuffer(uint32_t index) const;
    const ImageSubresourceSet& getReadSubresources(uint32_t index) const&;
    const ImageSubresourceSet& getWriteSubresources(uint32_t index) const&;

    void copyFrom(const ResourceLockerDescriptor* other);

    virtual uint32_t getImageCount() const = 0;
    virtual uint32_t getBufferCount() const = 0;
    virtual void clear() = 0;

    ResourceLockerDescriptor() = default;
    ResourceLockerDescriptor(const ResourceLockerDescriptor&) = delete;
    ResourceLockerDescriptor& operator=(const ResourceLockerDescriptor&) = delete;
    ResourceLockerDescriptor(ResourceLockerDescriptor&&) = default;
    ResourceLockerDescriptor& operator=(ResourceLockerDescriptor&&) = default;

 protected:
    ~ResourceLockerDescriptor() {}

    struct BufferData
    {
        drv::BufferPtr buffer = drv::get_null_ptr<drv::BufferPtr>();
        bool reads;
        bool writes;
        BufferData() : reads(false), writes(true) {}
        BufferData(drv::BufferPtr _buffer) : buffer(_buffer), reads(false), writes(false) {}
    };
    struct ImageData
    {
        drv::ImagePtr image = drv::get_null_ptr<drv::ImagePtr>();
        drv::ImageSubresourceSet reads;
        drv::ImageSubresourceSet writes;
        ImageData() : reads(0), writes(0) {}
        ImageData(drv::ImagePtr _image, uint32_t layerCount)
          : image(_image), reads(layerCount), writes(layerCount) {}
    };

    // TODO when buffer count is added, add it ot empty() function
    virtual void push_back(BufferData&& data) = 0;
    virtual void reserveBuffers(uint32_t count) = 0;

    virtual BufferData& getBufferData(uint32_t index) = 0;
    virtual const BufferData& getBufferData(uint32_t index) const = 0;

    virtual void push_back(ImageData&& data) = 0;
    virtual void reserveImages(uint32_t count) = 0;

    virtual ImageData& getImageData(uint32_t index) = 0;
    virtual const ImageData& getImageData(uint32_t index) const = 0;

 private:
    uint32_t findImage(ImagePtr image) const;
    uint32_t findBuffer(BufferPtr image) const;

    ImageSubresourceSet& getReadSubresources(uint32_t index) &;
    ImageSubresourceSet& getWriteSubresources(uint32_t index) &;
};

class ResourceLocker
{
 public:
    using LockId = uint32_t;

    class Lock
    {
     public:
        Lock() = default;
        Lock(ResourceLocker* _locker, const ResourceLockerDescriptor* _descriptor, LockId _lockId)
          : locker(_locker), descriptor(_descriptor), lockId(_lockId) {}

        Lock(const Lock&) = delete;
        Lock& operator=(const Lock&) = delete;
        Lock(Lock&& other)
          : locker(other.locker), descriptor(other.descriptor), lockId(other.lockId) {
            other.locker = nullptr;
        }
        Lock& operator=(Lock&& other) {
            if (this == &other)
                return *this;
            close();
            locker = other.locker;
            descriptor = other.descriptor;
            lockId = other.lockId;
            other.locker = nullptr;
            return *this;
        }

        friend class ResourceLocker;

        const ResourceLockerDescriptor* getDescriptor() const { return descriptor; }

        ~Lock() { close(); }

     private:
        ResourceLocker* locker = nullptr;
        const ResourceLockerDescriptor* descriptor = nullptr;
        LockId lockId;

        void close() {
            if (locker != nullptr) {
                locker->unlock(lockId);
                locker = nullptr;
            }
        }
    };

    template <typename R>
    class ResultLock
    {
     public:
        ResultLock(ResourceLocker* _locker, const ResourceLockerDescriptor* _descriptor,
                   LockId _lockId, R _result)
          : lock(_locker, _descriptor, _lockId), result(std::move(_result)) {}

        const R& get() const { return result; }
        operator R() const { return get(); }

        Lock&& getLock() && { return std::move(lock); }

     private:
        Lock lock;
        R result;
    };

    enum class LockResult
    {
        SUCCESS_IMMEDIATE,
        SUCCESS_BLOCKED
    };
    ResultLock<LockResult> lock(const ResourceLockerDescriptor* locker);

    enum class LockTimeResult
    {
        SUCCESS_IMMEDIATE,
        SUCCESS_BLOCKED,
        TIMEOUT
    };
    ResultLock<LockTimeResult> lockTimeout(const ResourceLockerDescriptor* locker,
                                           uint64_t timeoutNSec);

    enum class TryLockResult
    {
        SUCCESS,
        FAILURE
    };
    ResultLock<TryLockResult> tryLock(const ResourceLockerDescriptor* locker);

    ResourceLocker() = default;

    ResourceLocker(const ResourceLocker&) = delete;
    ResourceLocker& operator=(const ResourceLocker&) = delete;

    ~ResourceLocker();

 private:
    std::mutex mutex;
    std::condition_variable cv;
    uint64_t unlockCounter = 0;
    std::vector<const ResourceLockerDescriptor*> locks;
    LockId prevFree = 0;

    void unlock(LockId lockId);
    uint32_t getLockCount(const ResourceLockerDescriptor* lock) const;
};
}  // namespace drv
