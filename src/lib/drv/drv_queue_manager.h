#pragma once

#include <atomic>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include <drvtypes.h>
#include <string_hash.h>

#include "drvlane.h"

namespace drv
{
class QueueManager
{
 public:
    // struct QueueInfo
    // {
    //     QueuePtr queue = NULL_HANDLE;
    //     QueueFamilyPtr family = NULL_HANDLE;
    //     CommandTypeMask cmdTypeMask = 0;
    // };
    struct AcquireInfo
    {
        std::string commandLane;
        std::string commandQueue;
    };

    explicit QueueManager(PhysicalDevicePtr physicalDevice, LogicalDevicePtr device,
                          CommandLaneManager* commandLaneMgr);
    ~QueueManager();

    QueueManager(const QueueManager&) = delete;
    QueueManager& operator=(const QueueManager&) = delete;

    QueueManager(QueueManager&&) = delete;
    QueueManager& operator=(QueueManager&&) = delete;

    // QueueInfo acquire(const AcquireInfo& info);
    // void release(QueuePtr queue);

    // Acquires as many queues as possible, if it cannot acquire amount of queueCount
    // class QueueHandel
    // {
    //  public:
    //     enum SharingMode
    //     {
    //         EXCLUSIVE,
    //         SHARED
    //     };

    //     struct AcquireGroupInfo
    //     {
    //         struct Info
    //         {
    //             AcquireInfo acquireInfo;
    //             unsigned int queueCount = 1;
    //         };
    //         std::vector<Info> infos;
    //     };

    //     QueueHandel(QueueManager* mgr, StrHash group, SharingMode sharingMode,
    //                 const AcquireGroupInfo& info);

    //     ~QueueHandel();

    //     QueueHandel(const QueueHandel&) = delete;
    //     QueueHandel& operator=(const QueueHandel&) = delete;

    //     QueueHandel(QueueHandel&&);
    //     QueueHandel& operator=(QueueHandel&&);

    //     unsigned int count() const;
    //     QueueInfo get(unsigned int ind) const;
    //     QueueManager::QueueInfo operator[](unsigned int ind) const;
    //     CommandTypeMask getCommandTypeMask(unsigned int ind) const;
    //     QueuePtr getQueue(unsigned int ind) const;
    //     QueueFamilyPtr getFamily(unsigned int ind) const;

    //     void lock();
    //     void unlock();

    //     void setOwnership();

    //  private:
    //     QueueManager* mgr = nullptr;
    //     StrHash group;
    //     SharingMode sharingMode;
    //     std::thread::id ownerThread;
    //     std::vector<QueueInfo> queues;
    //     std::mutex mutex;
    //     std::unique_lock<std::mutex> uniqueLock;

    //     void checkThread() const;
    //     void close();
    // };

    // QueueHandel acquireHandel(QueueHandel::SharingMode sharingMode,
    //                           const QueueHandel::AcquireGroupInfo& info);
    // QueueHandel acquireHandel(QueueHandel::SharingMode sharingMode, const AcquireInfo& info,
    //                           unsigned int queueCount = 1);

    // bool available(const AcquireInfo& info) const;

 private:
    struct QueueInfo
    {
        CommandLaneManager::Queue info;
        drv::QueuePtr queue;
    };
    struct LaneInfo
    {
        std::unordered_map<std::string, QueueInfo> queues;
    };

    PhysicalDevicePtr physicalDevice;
    LogicalDevicePtr device;
    std::unordered_map<std::string, LaneInfo> lanes;
    std::mutex mutex;
};
}  // namespace drv
