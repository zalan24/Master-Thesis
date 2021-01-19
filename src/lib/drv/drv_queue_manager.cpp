#include "drv_queue_manager.h"

#include <algorithm>

#include <drverror.h>

#include "drv.h"

using namespace drv;

#define QM_SYNC std::unique_lock<std::mutex> lock(mutex)

// QueueManager::QueueHandel::QueueHandel(QueueManager* _mgr, StrHash _group, SharingMode _sharingMode,
//                                        const AcquireGroupInfo& groupInfo)
//   : mgr(_mgr),
//     group(_group),
//     sharingMode(_sharingMode),
//     ownerThread(std::this_thread::get_id()),
//     uniqueLock(mutex) {
//     unsigned int queueCount = 0;
//     for (const auto& info : groupInfo.infos)
//         queueCount = std::max(queueCount, info.queueCount);
//     for (unsigned int i = 0; i < queueCount; ++i) {
//         for (const auto& info : groupInfo.infos) {
//             if (i > info.queueCount)
//                 continue;
//             QueueInfo queueInfo = mgr->acquire(group, info.acquireInfo);
//             if (queueInfo.queue == nullptr)
//                 break;
//             queues.push_back(queueInfo);
//         }
//     }
//     uniqueLock.unlock();
// }

// QueueManager::QueueHandel::~QueueHandel() {
//     if (mgr == nullptr)
//         return;
//     close();
//     mgr = nullptr;
// }

// void QueueManager::QueueHandel::close() {
//     checkThread();
//     for (const QueueInfo& queue : queues)
//         mgr->release(group, queue.queue);
//     queues.clear();
// }

// QueueManager::QueueHandel::QueueHandel(QueueHandel&& other)
//   : mgr(other.mgr),
//     group(other.group),
//     sharingMode(other.sharingMode),
//     ownerThread(std::this_thread::get_id()),
//     queues(std::move(other.queues)),
//     uniqueLock(mutex) {
//     other.mgr = nullptr;
//     uniqueLock.unlock();
// }

// QueueManager::QueueHandel& QueueManager::QueueHandel::operator=(QueueHandel&& other) {
//     if (this == &other)
//         return *this;
//     close();
//     mgr = other.mgr;
//     group = other.group;
//     queues = std::move(other.queues);
//     other.mgr = nullptr;
//     return *this;
// }

// unsigned int QueueManager::QueueHandel::count() const {
//     return static_cast<unsigned int>(queues.size());
// }

// QueueManager::QueueInfo QueueManager::QueueHandel::get(unsigned int ind) const {
//     checkThread();
//     return queues[ind];
// }

// QueueManager::QueueInfo QueueManager::QueueHandel::operator[](unsigned int ind) const {
//     checkThread();
//     return queues[ind];
// }

// CommandTypeMask QueueManager::QueueHandel::getCommandTypeMask(unsigned int ind) const {
//     checkThread();
//     return queues[ind].cmdTypeMask;
// }

// QueuePtr QueueManager::QueueHandel::getQueue(unsigned int ind) const {
//     checkThread();
//     return queues[ind].queue;
// }

// QueueFamilyPtr QueueManager::QueueHandel::getFamily(unsigned int ind) const {
//     checkThread();
//     return queues[ind].family;
// }

// void QueueManager::QueueHandel::lock() {
//     if (sharingMode == SHARED)
//         uniqueLock.lock();
// }

// void QueueManager::QueueHandel::unlock() {
//     if (sharingMode == SHARED)
//         uniqueLock.unlock();
// }

// void QueueManager::QueueHandel::setOwnership() {
//     ownerThread = std::this_thread::get_id();
// }

// void QueueManager::QueueHandel::checkThread() const {
// #ifdef DEBUG
//     switch (sharingMode) {
//         case EXCLUSIVE:
//             drv_assert(std::this_thread::get_id() == ownerThread,
//                        "Exclusive QueueHandel can only be accessed from the owner thread");
//             break;
//         case SHARED:
//             drv_assert(uniqueLock.owns_lock(), "Shared QueueHandel must be locked before access");
//             break;
//     }
// #endif
// }

QueueManager::QueueManager(PhysicalDevicePtr _physicalDevice, LogicalDevicePtr _device,
                           CommandLaneManager* commandLaneMgr)
  : physicalDevice(_physicalDevice), device(_device) {
    for (const auto& itr : commandLaneMgr->getLanes()) {
        LaneInfo laneInfo;
        for (const auto& qItr : itr.second.queues) {
            QueueInfo qInfo;
            qInfo.info = qItr.second;
            qInfo.queue = drv::get_queue(device, qItr.second.familyPtr, qItr.second.queueIndex);
            drv_assert(qInfo.queue != NULL_HANDLE, "Could not get queue");
            laneInfo.queues[qItr.first] = std::move(qInfo);
        }
        lanes[itr.first] = std::move(laneInfo);
    }
}

QueueManager::~QueueManager() {
    QM_SYNC;
    // TODO
    // drv_assert(groups.size() == 0, "Not all queues were released");
}

// QueueManager::QueueInfo QueueManager::acquire(StrHash group, const AcquireInfo& info) {
//     QM_SYNC;
//     GroupInfo& groupInfo = groups[group];
//     QueueManager::QueueInfo ret;
//     ret.queue = NULL_HANDLE;
//     auto assignTo = [](auto& familyItr, QueuePtr queue, QueueInfo& _info) {
//         _info.queue = queue;
//         _info.family = familyItr.first;
//         _info.cmdTypeMask = familyItr.second.mask;
//     };
//     auto assign = [&](auto& familyItr, QueuePtr queue) { assignTo(familyItr, queue, ret); };
//     for (auto& itr : families) {
//         if ((itr.second.mask & info.requiredMask) != info.requiredMask)
//             continue;
//         if ((itr.second.mask & info.allowMask) != itr.second.mask)
//             continue;
//         for (QueuePtr& queue : itr.second.queues) {
//             if (groupInfo.used.count(queue))
//                 continue;
//             // First possible candidate
//             if (ret.queue == NULL_HANDLE) {
//                 assign(itr, queue);
//                 continue;
//             }
//             // Custom function
//             if (info.choose != nullptr) {
//                 QueueInfo current;
//                 assignTo(itr, queue, current);
//                 const QueueInfo* best = info.choose(current, ret);
//                 if (best != nullptr) {
//                     if (best == &current)
//                         ret = current;
//                     continue;
//                 }
//             }
//             // Preferenc
//             const unsigned int itrPreferMatch = mask_match_count(itr.second.mask, info.preferMask);
//             const unsigned int retPreferMatch = mask_match_count(ret.cmdTypeMask, info.preferMask);
//             if (itrPreferMatch != retPreferMatch) {
//                 if (itrPreferMatch < retPreferMatch)
//                     assign(itr, queue);
//                 continue;
//             }
//             // Priority
//             const float queuePriority = get_queue_info(device, queue).priority;
//             const float ret_queuePriority = get_queue_info(device, ret.queue).priority;
//             using std::abs;
//             if (info.targetPriority >= 0 && queuePriority != ret_queuePriority) {
//                 if (abs(info.targetPriority - queuePriority)
//                     < abs(info.targetPriority - ret_queuePriority))
//                     assign(itr, queue);
//                 continue;
//             }
//         }
//     }
//     groupInfo.used.insert(ret.queue);
//     return ret;
// }

// void QueueManager::release(StrHash group, QueuePtr queue) {
//     QM_SYNC;
//     auto itr = groups.find(group);
//     drv_assert(itr != groups.end(), "Queue is released from an unknown group");
//     drv_assert(itr->second.used.count(queue) > 0,
//                "Released queue was not acquired (or it has already been released)");
//     itr->second.used.erase(queue);
//     if (itr->second.used.size() == 0)
//         groups.erase(itr);
// }

// QueueManager::QueueHandel QueueManager::acquireHandel(StrHash group,
//                                                       QueueHandel::SharingMode sharingMode,
//                                                       const QueueHandel::AcquireGroupInfo& info) {
//     return QueueHandel{this, group, sharingMode, info};
// }

// QueueManager::QueueHandel QueueManager::acquireHandel(StrHash group,
//                                                       QueueHandel::SharingMode sharingMode,
//                                                       const AcquireInfo& info,
//                                                       unsigned int queueCount) {
//     QueueHandel::AcquireGroupInfo groupInfo;
//     groupInfo.infos.push_back({info, queueCount});
//     return QueueHandel{this, group, sharingMode, groupInfo};
// }

// bool QueueManager::available(StrHash group, const AcquireInfo& info) const {
//     unsigned int count = 0;
//     auto groupInfo = groups.find(group);
//     for (auto& itr : families) {
//         if ((itr.second.mask & info.requiredMask) != info.requiredMask)
//             continue;
//         if ((itr.second.mask & info.allowMask) != itr.second.mask)
//             continue;
//         for (const QueuePtr& queue : itr.second.queues) {
//             if (groupInfo != groups.end() && groupInfo->second.used.count(queue))
//                 continue;
//             count++;
//         }
//     }
//     return count;
// }
