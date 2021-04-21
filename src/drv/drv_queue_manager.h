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
    struct AcquireInfo
    {
        std::string commandLane;
        std::string commandQueue;
    };
    struct Queue
    {
        QueuePtr queue;
        CommandLaneManager::Queue info;
        Queue() : queue(get_null_ptr<QueuePtr>()) {}
    };

    explicit QueueManager(LogicalDevicePtr device, CommandLaneManager* commandLaneMgr);
    ~QueueManager();

    QueueManager(const QueueManager&) = delete;
    QueueManager& operator=(const QueueManager&) = delete;

    QueueManager(QueueManager&&) = delete;
    QueueManager& operator=(QueueManager&&) = delete;

    Queue getQueue(const AcquireInfo& info) const;

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

    LogicalDevicePtr device;
    std::unordered_map<std::string, LaneInfo> lanes;
    mutable std::mutex mutex;
};
}  // namespace drv
