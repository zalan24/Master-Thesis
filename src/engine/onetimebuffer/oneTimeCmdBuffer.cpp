// #include "oneTimeCmdBuffer.h"

// #include <drv_resource_tracker.h>

// #include <framegraph.h>

// OneTimeCmdBuffer::OneTimeCmdBuffer(std::unique_lock<std::mutex>&& _queueLock, drv::QueuePtr _queue,
//                                    FrameGraph::QueueId _queueId, FrameGraph* _frameGraph,
//                                    Engine* _engine, FrameGraph::NodeHandle* _nodeHandle,
//                                    FrameId _frameId,
//                                    drv::CommandBufferCirculator::CommandBufferHandle&& _cmdBuffer)
//   : queueLock(std::move(_queueLock)),
//     queue(_queue),
//     queueId(_queueId),
//     frameGraph(_frameGraph),
//     engine(_engine),
//     nodeHandle(_nodeHandle),
//     frameId(_frameId),
//     cmdBuffer(std::move(_cmdBuffer)),
//     resourceTracker(nodeHandle->getNode().getResourceTracker(queueId)),
//     signalSemaphores(engine->garbageSystem.getAllocator<decltype(signalSemaphores)::value_type>()),
//     signalTimelineSemaphores(
//       engine->garbageSystem.getAllocator<decltype(signalTimelineSemaphores)::value_type>()),
//     waitSemaphores(engine->garbageSystem.getAllocator<decltype(waitSemaphores)::value_type>()),
//     waitTimelineSemaphores(
//       engine->garbageSystem.getAllocator<decltype(waitTimelineSemaphores)::value_type>()) {
//     assert(cmdBuffer);
//     assert(
//       getResourceTracker()->begin_primary_command_buffer(cmdBuffer.commandBufferPtr, true, false));
//     nodeHandle->useQueue(queueId);
// }

// OneTimeCmdBuffer::OneTimeCmdBuffer(OneTimeCmdBuffer&& other)
//   : queueLock(std::move(other.queueLock)),
//     queue(other.queue),
//     queueId(other.queueId),
//     frameGraph(other.frameGraph),
//     engine(other.engine),
//     nodeHandle(other.nodeHandle),
//     frameId(other.frameId),
//     cmdBuffer(std::move(other.cmdBuffer)),
//     resourceTracker(other.resourceTracker),
//     signalSemaphores(std::move(other.signalSemaphores)),
//     signalTimelineSemaphores(std::move(other.signalTimelineSemaphores)),
//     waitSemaphores(std::move(other.waitSemaphores)),
//     waitTimelineSemaphores(std::move(other.waitTimelineSemaphores)) {
//     other.engine = nullptr;
// }

// OneTimeCmdBuffer& OneTimeCmdBuffer::operator=(OneTimeCmdBuffer&& other) {
//     if (&other == this)
//         return *this;
//     close();
//     queueLock = std::move(other.queueLock);
//     queue = other.queue;
//     queueId = other.queueId;
//     frameGraph = other.frameGraph;
//     engine = other.engine;
//     nodeHandle = other.nodeHandle;
//     frameId = other.frameId;
//     cmdBuffer = std::move(other.cmdBuffer);
//     resourceTracker = other.resourceTracker;
//     signalSemaphores = std::move(other.signalSemaphores);
//     signalTimelineSemaphores = std::move(other.signalTimelineSemaphores);
//     waitSemaphores = std::move(other.waitSemaphores);
//     waitTimelineSemaphores = std::move(other.waitTimelineSemaphores);
//     other.engine = nullptr;
//     return *this;
// }

// OneTimeCmdBuffer::~OneTimeCmdBuffer() {
//     close();
// }

// void OneTimeCmdBuffer::close() {
//     if (engine == nullptr)
//         return;
//     assert(getResourceTracker()->end_primary_command_buffer(cmdBuffer.commandBufferPtr));
//     ExecutionQueue* q = frameGraph->getExecutionQueue(*nodeHandle);
//     q->push(ExecutionPackage(ExecutionPackage::CommandBufferPackage{
//       queue, std::move(cmdBuffer), std::move(signalSemaphores), std::move(signalTimelineSemaphores),
//       std::move(waitSemaphores), std::move(waitTimelineSemaphores)}));
// }

// drv::CommandBufferPtr OneTimeCmdBuffer::acquireCommandBuffer() {
//     drv::CommandBufferBankGroupInfo acquireInfo(drv::get_queue_family(getDevice(), queue), false,
//                                                 drv::CommandBufferType::PRIMARY);
//     drv::drv_assert(!cmdBuffer);
//     cmdBuffer = bufferBank->acquire(acquireInfo);
//     return cmdBuffer.commandBufferPtr;
// }

// void OneTimeCmdBuffer::releaseCommandBuffer(drv::CommandBufferPtr cmdBuffer) {
//     drv::drv_assert(cmdBuffer.commandBufferPtr == cmdBuffer);
// }

// OneTimeCmdBuffer::OneTimeCmdBuffer(drv::CmdBufferBank* _bufferBank GarbageSystem* _garbageSystem)
//   : _bufferBank(_bufferBank), garbageSystem(_garbageSystem) {
// }
