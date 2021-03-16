#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <concurrentqueue.h>

#include <flexiblearray.h>

#include <drv.h>
#include <drv_queue_manager.h>
#include <drv_wrappers.h>
#include <drvbarrier.h>
#include <drvcmdbufferbank.h>
#include <drvlane.h>
#include <eventpool.h>
#include <framegraph.h>

// #include <entitymanager.h>
#include <input.h>
#include <inputmanager.h>
// #include <renderer.h>
#include <corecontext.h>
#include <resourcemanager.h>
#include <serializable.h>

#include "garbage.h"
#include "shaderbin.h"

struct ExecutionPackage;
class ISimulation;
class IRenderer;

class Engine
{
 public:
    struct Config final : public ISerializable
    {
        int screenWidth;
        int screenHeight;
        int imagesInSwapchain;
        int maxFramesInExecutionQueue;
        int maxFramesInFlight;
        std::string title;
        std::string driver;
        int inputBufferSize;
        int stackMemorySizeKb;
        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;
    };

    Engine(const Config& config, const std::string& shaderbinFile,
           ResourceManager::ResourceInfos resource_infos);
    Engine(const std::string& configFile, const std::string& shaderbinFile,
           ResourceManager::ResourceInfos resource_infos);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void initGame(IRenderer* renderer, ISimulation* simulation);
    void gameLoop();

    // EntityManager* getEntityManager() { return &entityManager; }
    // const EntityManager* getEntityManager() const { return &entityManager; }

    drv::LogicalDevicePtr getDevice() const { return device; }
    const ShaderBin* getShaderBin() const { return &shaderBin; }

    struct AcquiredImageData
    {
        drv::ImagePtr image;
        drv::SemaphorePtr imageAvailableSemaphore;
        drv::SemaphorePtr renderFinishedSemaphore;
    };
    AcquiredImageData acquiredSwapchainImage(FrameGraph::NodeHandle& acquiringNodeHandle);

    struct QueueData
    {
        drv::QueuePtr handle;
        FrameGraph::QueueId id;
    };

    struct QueueInfo
    {
        QueueData renderQueue;
        QueueData presentQueue;
        QueueData computeQueue;
        QueueData DtoHQueue;
        QueueData HtoDQueue;
        QueueData inputQueue;
    };
    const QueueInfo& getQueues() const;

    class CommandBufferRecorder
    {
     public:
        friend class Engine;

        CommandBufferRecorder(const CommandBufferRecorder&) = delete;
        CommandBufferRecorder& operator=(const CommandBufferRecorder&) = delete;
        CommandBufferRecorder(CommandBufferRecorder&& other);
        CommandBufferRecorder& operator=(CommandBufferRecorder&& other);

        ~CommandBufferRecorder();

        void cmdWaitSemaphore(drv::SemaphorePtr semaphore, drv::PipelineStages::FlagType waitStage);
        void cmdWaitTimelineSemaphore(drv::TimelineSemaphorePtr semaphore, uint64_t waitValue,
                                      drv::PipelineStages::FlagType waitStage);
        void cmdSignalSemaphore(drv::SemaphorePtr semaphore);
        void cmdSignalTimelineSemaphore(drv::TimelineSemaphorePtr semaphore, uint64_t signalValue);
        void cmdImageBarrier(const drv::ImageMemoryBarrier& barrier);
        void cmdClearImage(drv::ImagePtr image, const drv::ClearColorValue* clearColors,
                           uint32_t ranges = 0,
                           const drv::ImageSubresourceRange* subresourceRanges = nullptr);
        // These functions the same way as cmdImageBarrier, but it uses an event for sync
        // it could be a better option if the resource is needed a lot later
        void cmdEventBarrier(const drv::ImageMemoryBarrier& barrier);
        void cmdEventBarrier(uint32_t imageBarrierCount, const drv::ImageMemoryBarrier* barriers);

        void cmdWaitHostEvent(drv::EventPtr event, const drv::ImageMemoryBarrier& barrier);
        void cmdWaitHostEvent(drv::EventPtr event, uint32_t imageBarrierCount,
                              const drv::ImageMemoryBarrier* barriers);

        // allows nodes, that depend on the current node's gpu work (on current queue) to run after this submission completion
        void finishQueueWork();

     private:
        CommandBufferRecorder(std::unique_lock<std::mutex>&& queueLock, drv::QueuePtr queue,
                              FrameGraph::QueueId queueId, FrameGraph* frameGraph, Engine* engine,
                              FrameGraph::NodeHandle* nodeHandle, FrameGraph::FrameId frameId,
                              drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer);

        std::unique_lock<std::mutex> queueLock;
        drv::QueuePtr queue;
        FrameGraph::QueueId queueId;
        FrameGraph* frameGraph;
        Engine* engine;
        FrameGraph::NodeHandle* nodeHandle;
        FrameGraph::FrameId frameId;
        drv::CommandBufferCirculator::CommandBufferHandle cmdBuffer;

        // TODO frame mem allocator
        std::vector<ExecutionPackage::CommandBufferPackage::SemaphoreSignalInfo> signalSemaphores;
        std::vector<ExecutionPackage::CommandBufferPackage::TimelineSemaphoreSignalInfo>
          signalTimelineSemaphores;

        void close();
    };

    CommandBufferRecorder acquireCommandRecorder(FrameGraph::NodeHandle& acquiringNodeHandle,
                                                 FrameGraph::FrameId frameId,
                                                 FrameGraph::QueueId queueId);

 private:
    struct ErrorCallback
    {
        ErrorCallback();
    };
    struct WindowIniter
    {
        WindowIniter(IWindow* window, drv::InstancePtr instance);
        ~WindowIniter();
        IWindow* window;
    };
    struct SyncBlock
    {
        std::vector<drv::Semaphore> imageAvailableSemaphores;
        std::vector<drv::Semaphore> renderFinishedSemaphores;
        SyncBlock(drv::LogicalDevicePtr device, uint32_t maxFramesInFlight);
    };

    Config config;

    ErrorCallback callback;
    Input input;
    InputManager inputManager;
    CoreContext coreContext;
    drv::DriverWrapper driver;
    drv::Window window;
    drv::Instance drvInstance;
    WindowIniter windowIniter;
    drv::DeviceExtensions deviceExtensions;
    drv::PhysicalDevice physicalDevice;
    drv::CommandLaneManager commandLaneMgr;
    drv::LogicalDevice device;
    drv::QueueManager queueManager;
    drv::QueueManager::Queue renderQueue;
    drv::QueueManager::Queue presentQueue;
    drv::QueueManager::Queue computeQueue;
    drv::QueueManager::Queue DtoHQueue;
    drv::QueueManager::Queue HtoDQueue;
    drv::QueueManager::Queue inputQueue;
    drv::CommandBufferBank cmdBufferBank;
    drv::Swapchain swapchain;
    EventPool eventPool;
    SyncBlock syncBlock;
    ShaderBin shaderBin;
    ResourceManager resourceMgr;
    // EntityManager entityManager;
    FrameGraph frameGraph;

    ISimulation* simulation = nullptr;
    IRenderer* renderer = nullptr;
    FrameGraph::NodeId inputSampleNode;
    FrameGraph::NodeId simStartNode;
    FrameGraph::NodeId simEndNode;
    FrameGraph::NodeId recordStartNode;
    FrameGraph::NodeId recordEndNode;
    FrameGraph::NodeId executeStartNode;
    FrameGraph::NodeId executeEndNode;
    FrameGraph::NodeId presentFrameNode;
    FrameGraph::NodeId cleanUpNode;
    QueueInfo queueInfos;

    mutable std::mutex garbageMutex;
    std::atomic<uint32_t> currentGarbage = 0;
    std::atomic<uint32_t> oldestGarbage = 0;
    flexiblearray<Garbage, 8> trashBins;

    template <typename F>
    void useGarbage(F&& f) {
        std::unique_lock<std::mutex> lock(garbageMutex);
        f(&trashBins[currentGarbage.load() % trashBins.size()]);
    }

    uint32_t acquireImageSemaphoreId = 0;

    mutable std::shared_mutex stopFrameMutex;
    mutable std::mutex executionMutex;

    void simulationLoop(volatile std::atomic<FrameGraph::FrameId>* simulationFrame,
                        const volatile std::atomic<FrameGraph::FrameId>* stopFrame);
    void recordCommandsLoop(const volatile std::atomic<FrameGraph::FrameId>* stopFrame);
    void executeCommandsLoop();
    void cleanUpLoop(const volatile std::atomic<FrameGraph::FrameId>* stopFrame);
    bool execute(FrameGraph::FrameId& executionFrame, ExecutionPackage&& package);
    void present(FrameGraph::FrameId presentFrame);
    bool sampleInput(FrameGraph::FrameId frameId);

    static drv::PhysicalDevice::SelectionInfo get_device_selection_info(
      drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions);
    static drv::Swapchain::CreateInfo get_swapchain_create_info(const Config& config);
};
