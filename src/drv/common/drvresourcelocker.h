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

 private:
};

class ResourceLocker
{
 public:
    enum LockResult
    {
        SUCCESS_IMMEDIATE,
        SUCCESS_BLOCKED
    };
    LockResult lock(const ResourceLockerDescriptor& locker);

    enum LockTimeResult
    {
        SUCCESS_IMMEDIATE,
        SUCCESS_BLOCKED,
        TIMEOUT
    };
    LockTimeResult lockTimeout(const ResourceLockerDescriptor& locker, uint64_t timeoutNSec);

    enum TryLockResult
    {
        SUCCESS,
        FAILURE
    };
    TryLockResult tryLock(const ResourceLockerDescriptor& locker);

    TODO;  // unlock ???
           // struct LockerObejct
           // {};

 private:
    std::mutex mutex;
    std::condition_variable cv;
    uint64_t unlockCounter = 0;
};
}  // namespace drv
