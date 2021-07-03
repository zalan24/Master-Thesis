#pragma once

#include <condition_variable>
#include <mutex>

#include <drvimage_types.h>
#include <drvresourceptrs.hpp>

namespace drv
{
class ResourceLocker;

class ResourceLockerDescriptor
{
 public:
    explicit ResourceLockerDescriptor() {}
    explicit ResourceLockerDescriptor(ResourceLocker* resource_locker);

    enum UsageMode
    {
        NONE = 0,
        READ = 1,
        WRITE = 2,
        READ_WRITE = 3
    };

    void addImage(drv::ImagePtr image, uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect,
                  UsageMode usage);
    void addImage(drv::ImagePtr image, const drv::ImageSubresourceSet& subresources,
                  UsageMode usage);
    void addImage(drv::ImagePtr image, UsageMode usage);  // all subresources

    UsageMode getImageUsage(drv::ImagePtr image, uint32_t layer, uint32_t mip,
                            drv::AspectFlagBits aspect) const;

    void clear();

    enum ConflictType
    {
        NONE_NO_OVERLAP,
        NONE_COMMON_READ,
        CONFLICT_WRITE
    };

    ConflictType findConflict(const ResourceLockerDescriptor& other) const;

    bool empty() const;
    uint32_t getImageCount() const;
    ImagePtr getImage(uint32_t index) const;
    const ImageSubresourceSet& getReadSubresources(uint32_t index) const&;
    const ImageSubresourceSet& getWriteSubresources(uint32_t index) const&;

 private:
};

class ResourceLocker
{
 public:
    using LockId = uint32_t;

    class Lock
    {
     public:
        Lock() = default;
        Lock(ResourceLocker* _locker, LockId _lockId) : locker(_locker), lockId(_lockId) {}

        Lock(const Lock&) = delete;
        Lock& operator=(const Lock&) = delete;
        Lock(Lock&& other) : locker(other.locker), lockId(other.lockId) { other.locker = nullptr; }
        Lock& operator=(Lock&& other) {
            if (this == &other)
                return *this;
            close();
            locker = other.locker;
            lockId = other.lockId;
            other.locker = nullptr;
            return *this;
        }

        friend class ResourceLocker;

        ~Lock() { close(); }

     private:
        ResourceLocker* locker = nullptr;
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
        ResultLock(ResourceLocker* _locker, LockId _lockId, R _result)
          : lock(_locker, _lockId), result(std::move(_result)) {}

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
    ResultLock<LockResult> lock(const ResourceLockerDescriptor& locker);

    enum class LockTimeResult
    {
        SUCCESS_IMMEDIATE,
        SUCCESS_BLOCKED,
        TIMEOUT
    };
    ResultLock<LockTimeResult> lockTimeout(const ResourceLockerDescriptor& locker,
                                           uint64_t timeoutNSec);

    enum class TryLockResult
    {
        SUCCESS,
        FAILURE
    };
    ResultLock<TryLockResult> tryLock(const ResourceLockerDescriptor& locker);

 private:
    std::mutex mutex;
    std::condition_variable cv;
    uint64_t unlockCounter = 0;

    void unlock(LockId lockId);
};
}  // namespace drv
