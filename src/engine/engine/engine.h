#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <concurrentqueue.h>

#include <logger.h>

#include <drv.h>
#include <drv_queue_manager.h>
#include <drv_wrappers.h>
#include <drvbarrier.h>
#include <drvcmdbufferbank.h>
#include <drvlane.h>

#include <eventpool.h>
#include <framegraph.h>

#include <corecontext.h>
#include <garbagesystem.h>
#include <input.h>
#include <inputmanager.h>
#include <resourcemanager.h>
#include <serializable.h>

#include <runtimestats.h>
#include <shaderbin.h>
#include <cmdBuffer.hpp>
#include <oneTimeCmdBuffer.hpp>

struct ExecutionPackage;

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
        int frameMemorySizeKb;
        drv::StateTrackingConfig trackerConfig;
        std::string logs;
        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;
    };

    struct Args
    {
        bool renderdocEnabled;
        bool gfxCaptureEnabled;
        bool apiDumpEnabled;
        std::string runtimeStatsBin;
    };

    Engine(int argc, char* argv[], const Config& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           ResourceManager::ResourceInfos resource_infos, const Args& args);
    virtual ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void gameLoop();

    // TODO remove
    virtual void record(FrameId frameId) = 0;
    virtual void simulate(FrameId frameId) = 0;
    FrameGraph::NodeId getRecStartNode() const { return recordStartNode; }
    FrameGraph::NodeId getRecEndNode() const { return recordEndNode; }
    // ---

    drv::LogicalDevicePtr getDevice() const { return device; }
    drv::PhysicalDevicePtr getPhysicalDevice() const { return physicalDevice; }
    const ShaderBin* getShaderBin() const { return &shaderBin; }

    using SwapchaingVersion = uint64_t;
    static constexpr SwapchaingVersion INVALID_SWAPCHAIN =
      std::numeric_limits<SwapchaingVersion>::max();
    struct AcquiredImageData
    {
        drv::ImagePtr image;
        uint32_t imageIndex;
        drv::SemaphorePtr imageAvailableSemaphore;
        drv::SemaphorePtr renderFinishedSemaphore;
        drv::Extent2D extent;
        SwapchaingVersion version;  // incremented upon recreation
        uint32_t imageCount;
        const drv::ImagePtr* images = nullptr;
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

    // OneTimeCmdBuffer acquireCommandRecorder(FrameGraph::NodeHandle& acquiringNodeHandle,
    //                                         FrameId frameId, FrameGraph::QueueId queueId);

    GarbageSystem* getGarbageSystem() { return &garbageSystem; }
    drv::CommandBufferBank* getCommandBufferBank() { return &cmdBufferBank; }

 protected:
    // Needs to be called from game implementation after finishing the framegraph
    void buildFrameGraph(FrameGraph::NodeId presentDepNode, FrameGraph::QueueId depQueueId);

    FrameGraph& getFrameGraph() { return frameGraph; }
    const FrameGraph& getFrameGraph() const { return frameGraph; }

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

    Logger logger;
    ErrorCallback callback;
    CoreContext coreContext;
    ShaderBin shaderBin;
    Input input;
    InputManager inputManager;
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
    ResourceManager resourceMgr;
    // EntityManager entityManager;
    GarbageSystem garbageSystem;
    FrameGraph frameGraph;
    RuntimeStats runtimeStats;

    FrameGraph::NodeId inputSampleNode;
    FrameGraph::TagNodeId simStartNode;
    FrameGraph::TagNodeId simEndNode;
    FrameGraph::NodeId recordStartNode;
    FrameGraph::NodeId recordEndNode;
    FrameGraph::TagNodeId executeStartNode;
    FrameGraph::TagNodeId executeEndNode;
    FrameGraph::NodeId presentFrameNode;
    FrameGraph::NodeId cleanUpNode;
    QueueInfo queueInfos;

    uint32_t acquireImageSemaphoreId = 0;
    SwapchaingVersion swapchainVersion = 0;

    mutable std::shared_mutex stopFrameMutex;
    mutable std::mutex executionMutex;

    void simulationLoop(volatile std::atomic<FrameId>* simulationFrame,
                        const volatile std::atomic<FrameId>* stopFrame);
    void recordCommandsLoop(const volatile std::atomic<FrameId>* stopFrame);
    void executeCommandsLoop();
    void cleanUpLoop(const volatile std::atomic<FrameId>* stopFrame);
    bool execute(FrameId& executionFrame, ExecutionPackage&& package);
    void present(FrameId presentFrame, uint32_t semaphoreIndex);
    bool sampleInput(FrameId frameId);

    static drv::PhysicalDevice::SelectionInfo get_device_selection_info(
      drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions,
      const ShaderBin& shaderBin);
    static drv::Swapchain::CreateInfo get_swapchain_create_info(const Config& config,
                                                                drv::QueuePtr present_queue,
                                                                drv::QueuePtr render_queue);
};
