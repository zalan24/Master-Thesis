#include "drvresourcelocker.h"

#include <chrono>

#include <drverror.h>

using namespace drv;

ResourceLocker::~ResourceLocker() {
    std::unique_lock<std::mutex> lock(mutex);
    bool allNull = true;
    for (const auto& itr : locks)
        if (itr != nullptr)
            allNull = false;
    drv::drv_assert(allNull, "Not all resource locks have been released");
}

uint32_t ResourceLocker::getLockCount(const ResourceLockerDescriptor* locker) const {
    uint32_t lockCount = 0;
    for (const auto& itr : locks) {
        if (itr == nullptr)
            continue;
        if (itr->findConflict(locker) == ResourceLockerDescriptor::CONFLICT_WRITE)
            lockCount++;
    }
    return lockCount;
}

ResourceLocker::ResultLock<ResourceLocker::LockResult> ResourceLocker::lock(
  const ResourceLockerDescriptor* locker) {
    std::unique_lock<std::mutex> lock(mutex);
    uint32_t lockCount = getLockCount(locker);
    LockResult ret = lockCount > 0 ? LockResult::SUCCESS_BLOCKED : LockResult::SUCCESS_IMMEDIATE;
    while (lockCount > 0) {
        uint64_t originalCounter = unlockCounter;  // if lockCount overflows
        uint64_t requiredUnlockCount = unlockCounter + lockCount;
        cv.wait(lock, [&, this] {
            return unlockCounter >= requiredUnlockCount || unlockCounter < originalCounter;
        });
        lockCount = getLockCount(locker);
    }
    if (prevFree >= locks.size() || locks[prevFree] != nullptr) {
        prevFree = 0;
        while (prevFree < locks.size() && locks[prevFree] == nullptr)
            ++prevFree;
    }
    if (prevFree == locks.size())
        locks.push_back(nullptr);  // not prevFree is valid
    locks[prevFree] = locker;
    return ResultLock<LockResult>(this, locker, prevFree, ret);
}

ResourceLocker::ResultLock<ResourceLocker::LockTimeResult> ResourceLocker::lockTimeout(
  const ResourceLockerDescriptor* locker, uint64_t timeoutNSec) {
    std::chrono::high_resolution_clock::time_point startTime =
      std::chrono::high_resolution_clock::now();
    std::unique_lock<std::mutex> lock(mutex);
    uint32_t lockCount = getLockCount(locker);
    LockTimeResult ret =
      lockCount > 0 ? LockTimeResult::SUCCESS_BLOCKED : LockTimeResult::SUCCESS_IMMEDIATE;
    while (lockCount > 0) {
        std::chrono::nanoseconds dur = std::chrono::high_resolution_clock::now() - startTime;
        if (uint64_t(dur.count()) >= timeoutNSec)
            return ResultLock<LockTimeResult>(this, nullptr, 0, LockTimeResult::TIMEOUT);
        std::chrono::nanoseconds durLeft = std::chrono::nanoseconds(timeoutNSec) - dur;
        uint64_t originalCount = unlockCounter;  // if lockCount overflows
        uint64_t requiredUnlockCount = unlockCounter + lockCount;
        cv.wait_for(lock, durLeft, [&, this] {
            return unlockCounter >= requiredUnlockCount || lockCount < originalCount;
        });
        lockCount = getLockCount(locker);
    }
    if (prevFree >= locks.size() || locks[prevFree] != nullptr) {
        prevFree = 0;
        while (prevFree < locks.size() && locks[prevFree] == nullptr)
            ++prevFree;
    }
    if (prevFree == locks.size())
        locks.push_back(nullptr);  // not prevFree is valid
    locks[prevFree] = locker;
    return ResultLock<LockTimeResult>(this, locker, prevFree, ret);
}

ResourceLocker::ResultLock<ResourceLocker::TryLockResult> ResourceLocker::tryLock(
  const ResourceLockerDescriptor* locker) {
    std::unique_lock<std::mutex> lock(mutex);
    uint32_t lockCount = getLockCount(locker);
    if (lockCount > 0)
        return ResultLock<TryLockResult>(this, nullptr, 0, TryLockResult::FAILURE);
    if (prevFree >= locks.size() || locks[prevFree] != nullptr) {
        prevFree = 0;
        while (prevFree < locks.size() && locks[prevFree] == nullptr)
            ++prevFree;
    }
    if (prevFree == locks.size())
        locks.push_back(nullptr);  // not prevFree is valid
    locks[prevFree] = locker;
    return ResultLock<TryLockResult>(this, locker, prevFree, TryLockResult::SUCCESS);
}

void ResourceLocker::unlock(LockId lockId) {
    std::unique_lock<std::mutex> lock(mutex);
    locks[lockId] = nullptr;
    prevFree = lockId;
    unlockCounter++;
    cv.notify_all();
}

void ResourceLockerDescriptor::addImage(drv::ImagePtr image, uint32_t layerCount, uint32_t layer,
                                        uint32_t mip, drv::AspectFlagBits aspect,
                                        UsageMode usage) noexcept {
    if (usage == UsageMode::NONE)
        return;
    uint32_t ind = findImage(image);
    if (ind != getImageCount()) {
        if (usage == UsageMode::READ || usage == UsageMode::READ_WRITE)
            getReadSubresources(ind).add(layer, mip, aspect);
        if (usage == UsageMode::WRITE || usage == UsageMode::READ_WRITE)
            getWriteSubresources(ind).add(layer, mip, aspect);
    }
    else {
        ind = getImageCount();
        push_back(ImageData{});
        while (0 < ind && image < getImage(ind - 1)) {
            getImageData(ind) = std::move(getImageData(ind - 1));
            ind--;
        }
        ImageData data(image, layerCount);
        if (usage == UsageMode::READ || usage == UsageMode::READ_WRITE)
            data.reads.add(layer, mip, aspect);
        if (usage == UsageMode::WRITE || usage == UsageMode::READ_WRITE)
            data.writes.add(layer, mip, aspect);
        getImageData(ind) = std::move(data);
    }
}

void ResourceLockerDescriptor::addBuffer(drv::BufferPtr buffer, UsageMode usage) noexcept {
    if (usage == UsageMode::NONE)
        return;
    uint32_t ind = findBuffer(buffer);
    if (ind != getBufferCount()) {
        if (usage == UsageMode::READ || usage == UsageMode::READ_WRITE)
            getBufferData(ind).reads = true;
        if (usage == UsageMode::WRITE || usage == UsageMode::READ_WRITE)
            getBufferData(ind).writes = true;
    }
    else {
        ind = getBufferCount();
        push_back(BufferData{});
        while (0 < ind && buffer < getBuffer(ind - 1)) {
            getBufferData(ind) = std::move(getBufferData(ind - 1));
            ind--;
        }
        BufferData data(buffer);
        if (usage == UsageMode::READ || usage == UsageMode::READ_WRITE)
            data.reads = true;
        if (usage == UsageMode::WRITE || usage == UsageMode::READ_WRITE)
            data.writes = true;
        getBufferData(ind) = std::move(data);
    }
}

void ResourceLockerDescriptor::addImage(drv::ImagePtr image,
                                        const drv::ImageSubresourceSet& subresources,
                                        UsageMode usage) noexcept {
    if (usage == UsageMode::NONE)
        return;
    uint32_t ind = findImage(image);
    if (ind != getImageCount()) {
        if (usage == UsageMode::READ || usage == UsageMode::READ_WRITE)
            getReadSubresources(ind).merge(subresources);
        if (usage == UsageMode::WRITE || usage == UsageMode::READ_WRITE)
            getWriteSubresources(ind).merge(subresources);
    }
    else {
        ind = getImageCount();
        push_back(ImageData{});
        while (0 < ind && image < getImage(ind - 1)) {
            getImageData(ind) = std::move(getImageData(ind - 1));
            ind--;
        }
        ImageData data(image, subresources.getLayerCount());
        if (usage == UsageMode::READ || usage == UsageMode::READ_WRITE)
            data.reads = subresources;
        if (usage == UsageMode::WRITE || usage == UsageMode::READ_WRITE)
            data.writes = subresources;
        getImageData(ind) = std::move(data);
    }
}

uint32_t ResourceLockerDescriptor::findImage(ImagePtr image) const {
    uint32_t a = 0;
    uint32_t b = getImageCount();
    if (b == 0)
        return getImageCount();
    while (a + 1 < b) {
        uint32_t m = (a + b) / 2;
        if (image < getImage(m))
            b = m;
        else
            a = m;
    }
    if (getImage(a) != image)
        return getImageCount();
    return a;
}

void ResourceLockerDescriptor::copyFrom(const ResourceLockerDescriptor* other) {
    clear();
    uint32_t imageCount = other->getImageCount();
    reserveImages(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i)
        push_back(ImageData(other->getImageData(i)));
    uint32_t bufferCount = other->getBufferCount();
    reserveImages(bufferCount);
    for (uint32_t i = 0; i < bufferCount; ++i)
        push_back(BufferData(other->getBufferData(i)));
}

ResourceLockerDescriptor::UsageMode ResourceLockerDescriptor::getImageUsage(
  uint32_t index, uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) const {
    bool read = getReadSubresources(index).has(layer, mip, aspect);
    bool write = getWriteSubresources(index).has(layer, mip, aspect);
    if (!read && !write)
        return UsageMode::NONE;
    if (!write)
        return UsageMode::READ;
    if (!read)
        return UsageMode::WRITE;
    return UsageMode::READ_WRITE;
}

ResourceLockerDescriptor::UsageMode ResourceLockerDescriptor::getImageUsage(
  drv::ImagePtr image, uint32_t layer, uint32_t mip, drv::AspectFlagBits aspect) const {
    uint32_t ind = findImage(image);
    if (ind == getImageCount())
        return UsageMode::NONE;
    return getImageUsage(ind, layer, mip, aspect);
}

ResourceLockerDescriptor::ConflictType ResourceLockerDescriptor::findConflict(
  const ResourceLockerDescriptor* other) const {
    uint32_t count1 = getImageCount();
    uint32_t count2 = other->getImageCount();
    uint32_t ind1 = 0;
    uint32_t ind2 = 0;
    ConflictType ret = ConflictType::NONE_NO_OVERLAP;
    while (ind1 < count1 && ind2 < count2) {
        if (getImage(ind1) < other->getImage(ind2)) {
            ind1++;
        }
        else if (other->getImage(ind2) < getImage(ind1)) {
            ind2++;
        }
        else {
            // equal
            if (getWriteSubresources(ind1).overlap(other->getWriteSubresources(ind2)))
                return ConflictType::CONFLICT_WRITE;
            if (getWriteSubresources(ind1).overlap(other->getReadSubresources(ind2)))
                return ConflictType::CONFLICT_WRITE;
            if (getReadSubresources(ind1).overlap(other->getWriteSubresources(ind2)))
                return ConflictType::CONFLICT_WRITE;
            if (getReadSubresources(ind1).overlap(other->getReadSubresources(ind2)))
                ret = ConflictType::NONE_COMMON_READ;
            ind1++;
            ind2++;
        }
    }
    return ret;
}

bool ResourceLockerDescriptor::empty() const {
    return getImageCount() == 0 && getBufferCount() == 0;
}

ImagePtr ResourceLockerDescriptor::getImage(uint32_t index) const {
    return getImageData(index).image;
}

const ImageSubresourceSet& ResourceLockerDescriptor::getReadSubresources(uint32_t index) const& {
    return getImageData(index).reads;
}

const ImageSubresourceSet& ResourceLockerDescriptor::getWriteSubresources(uint32_t index) const& {
    return getImageData(index).writes;
}

ImageSubresourceSet& ResourceLockerDescriptor::getReadSubresources(uint32_t index) & {
    return getImageData(index).reads;
}

ImageSubresourceSet& ResourceLockerDescriptor::getWriteSubresources(uint32_t index) & {
    return getImageData(index).writes;
}
