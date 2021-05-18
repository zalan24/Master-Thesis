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
        uint32_t screenWidth;
        uint32_t screenHeight;
        uint32_t imagesInSwapchain;
        uint32_t maxFramesInExecutionQueue;
        uint32_t maxFramesInFlight;
        std::string title;
        std::string driver;
        uint32_t inputBufferSize;
        uint32_t stackMemorySizeKb;
        uint32_t frameMemorySizeKb;
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

    drv::LogicalDevicePtr getDevice() const { return device; }
    drv::PhysicalDevicePtr getPhysicalDevice() const { return physicalDevice; }
    const ShaderBin* getShaderBin() const { return &shaderBin; }

    using SwapchaingVersion = uint64_t;
    static constexpr SwapchaingVersion INVALID_SWAPCHAIN =
      std::numeric_limits<SwapchaingVersion>::max();
    struct AcquiredImageData
    {
        drv::ImagePtr image = drv::get_null_ptr<drv::ImagePtr>();
        uint32_t imageIndex;
        uint32_t semaphoreIndex;
        drv::SemaphorePtr imageAvailableSemaphore;
        drv::SemaphorePtr renderFinishedSemaphore;
        drv::Extent2D extent;
        SwapchaingVersion version;  // incremented upon recreation
        uint32_t imageCount = 0;
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

    virtual void simulate(FrameId frameId) = 0;
    virtual void beforeDraw(FrameId frameId) = 0;
    virtual AcquiredImageData record(FrameId frameId) = 0;
    virtual void readback(FrameId frameId) = 0;

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
    FrameGraph::NodeId presentFrameNode;
    QueueInfo queueInfos;

    uint32_t acquireImageSemaphoreId = 0;
    SwapchaingVersion swapchainVersion = 0;

    mutable std::mutex executionMutex;

    void simulationLoop();
    void beforeDrawLoop();
    void recordCommandsLoop();
    void executeCommandsLoop();
    void readbackLoop();
    bool execute(ExecutionPackage&& package);
    void present(uint32_t imageIndex, uint32_t semaphoreIndex);
    bool sampleInput(FrameId frameId);

    static drv::PhysicalDevice::SelectionInfo get_device_selection_info(
      drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions,
      const ShaderBin& shaderBin);
    static drv::Swapchain::CreateInfo get_swapchain_create_info(const Config& config,
                                                                drv::QueuePtr present_queue,
                                                                drv::QueuePtr render_queue);
};
