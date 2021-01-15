#pragma once

#include <atomic>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include <drvtypes.h>
#include <string_hash.h>

namespace drv
{
class QueueManager
{
 public:
    struct QueueInfo
    {
        QueuePtr queue = NULL_HANDLE;
        QueueFamilyPtr family = NULL_HANDLE;
        CommandTypeMask cmdTypeMask = 0;
    };
    struct AcquireInfo
    {
        CommandTypeMask requiredMask = 0;
        CommandTypeMask allowMask = CommandTypeBits::CMD_TYPE_ALL;
        // These selectors will be applied in the same order as they are declared
        // default values don't affect selection
        // returns pointer to the better one or nullptr if they are equally good
        const QueueInfo* (*choose)(const QueueInfo&, const QueueInfo&) = nullptr;
        CommandTypeMask preferMask = 0;  // pick the closest to prefer mask
        float targetPriority = -1;       // -1 for don't care
    };

    explicit QueueManager(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device);
    ~QueueManager();

    QueueManager(const QueueManager&) = delete;
    QueueManager& operator=(const QueueManager&) = delete;

    QueueManager(QueueManager&&) = delete;
    QueueManager& operator=(QueueManager&&) = delete;

    QueueInfo acquire(StrHash group, const AcquireInfo& info);
    void release(StrHash group, QueuePtr queue);

    // Acquires as many queues as possible, if it cannot acquire amount of queueCount
    class QueueHandel
    {
     public:
        enum SharingMode
        {
            EXCLUSIVE,
            SHARED
        };

        struct AcquireGroupInfo
        {
            struct Info
            {
                AcquireInfo acquireInfo;
                unsigned int queueCount = 1;
            };
            std::vector<Info> infos;
        };

        QueueHandel(QueueManager* mgr, StrHash group, SharingMode sharingMode,
                    const AcquireGroupInfo& info);

        ~QueueHandel();

        QueueHandel(const QueueHandel&) = delete;
        QueueHandel& operator=(const QueueHandel&) = delete;

        QueueHandel(QueueHandel&&);
        QueueHandel& operator=(QueueHandel&&);

        unsigned int count() const;
        QueueInfo get(unsigned int ind) const;
        QueueManager::QueueInfo operator[](unsigned int ind) const;
        CommandTypeMask getCommandTypeMask(unsigned int ind) const;
        QueuePtr getQueue(unsigned int ind) const;
        QueueFamilyPtr getFamily(unsigned int ind) const;

        void lock();
        void unlock();

        void setOwnership();

     private:
        QueueManager* mgr = nullptr;
        StrHash group;
        SharingMode sharingMode;
        std::thread::id ownerThread;
        std::vector<QueueInfo> queues;
        std::mutex mutex;
        std::unique_lock<std::mutex> uniqueLock;

        void checkThread() const;
        void close();
    };

    QueueHandel acquireHandel(StrHash group, QueueHandel::SharingMode sharingMode,
                              const QueueHandel::AcquireGroupInfo& info);
    QueueHandel acquireHandel(StrHash group, QueueHandel::SharingMode sharingMode,
                              const AcquireInfo& info, unsigned int queueCount = 1);

    unsigned int available(StrHash group, const AcquireInfo& info) const;

 private:
    struct FamilyInfo
    {
        CommandTypeMask mask;
        std::vector<QueuePtr> queues;
    };
    struct GroupInfo
    {
        std::set<QueuePtr> used;
    };

    PhysicalDevicePtr physicalDevice;
    LogicalDevicePtr device;
    std::unordered_map<QueueFamilyPtr, FamilyInfo> families;
    std::unordered_map<StrHash, GroupInfo> groups;
    std::mutex mutex;
};
}  // namespace drv
