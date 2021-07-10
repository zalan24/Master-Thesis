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
#include <drvresourcelocker.h>
#include <drvsemaphorepool.h>

#include <eventpool.h>
#include <framegraph.h>

#include <corecontext.h>
#include <garbagesystem.h>
#include <input.h>
#include <inputmanager.h>
#include <serializable.h>

#include <runtimestats.h>
#include <shaderbin.h>
#include <cmdBuffer.hpp>
#include <oneTimeCmdBuffer.hpp>

#include "resources.hpp"

struct ExecutionPackage;

struct EngineConfig final : public IAutoSerializable<EngineConfig>
{
    REFLECTABLE
    (
        (uint32_t) screenWidth,
        (uint32_t) screenHeight,
        (uint32_t) imagesInSwapchain,
        (uint32_t) maxFramesInExecutionQueue,
        (uint32_t) maxFramesInFlight,
        (std::string) title,
        (std::string) driver,
        (uint32_t) inputBufferSize,
        (uint32_t) stackMemorySizeKb,
        (uint32_t) frameMemorySizeKb,
        (std::string) logs
    )
};

class Engine
{
 public:
    struct Args
    {
        bool renderdocEnabled = false;
        bool gfxCaptureEnabled = false;
        bool apiDumpEnabled = false;
        std::string runtimeStatsPersistanceBin;
        std::string runtimeStatsGameExportsBin;
        std::string runtimeStatsCacheBin;
        std::string reportFile;
        bool clearRuntimeStats = false;
    };

    Engine(int argc, char* argv[], const EngineConfig& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           Args args);
    virtual ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void gameLoop();

    drv::LogicalDevicePtr getDevice() const { return device; }
    drv::PhysicalDevicePtr getPhysicalDevice() const { return physicalDevice; }
    const ShaderBin* getShaderBin() const { return &shaderBin; }
    struct AcquiredImageData
    {
        drv::ImagePtr image = drv::get_null_ptr<drv::ImagePtr>();
        drv::SwapchainPtr swapchain = drv::get_null_ptr<drv::SwapchainPtr>();
        uint32_t imageIndex = drv::Swapchain::INVALID_INDEX;
        uint32_t semaphoreIndex = 0;
        drv::SemaphorePtr imageAvailableSemaphore = drv::get_null_ptr<drv::SemaphorePtr>();
        drv::SemaphorePtr renderFinishedSemaphore = drv::get_null_ptr<drv::SemaphorePtr>();
        drv::Extent2D extent = {0, 0};
        uint32_t imageCount = 0;
        const drv::ImagePtr* images = nullptr;
        operator bool() const { return !drv::is_null_ptr(image); }
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

    template <typename T, typename... Args>
    auto createResource(Args&&... args) {
        return res::GarbageResource<T>(getGarbageSystem(), std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    void createResource(res::GarbageResource<T>& resource, Args&&... args) {
        resource = res::GarbageResource<T>(getGarbageSystem(), std::forward<Args>(args)...);
    }

    uint32_t getMaxFramesInFlight() const;

 protected:
    // Needs to be called from game implementation after finishing the framegraph
    void buildFrameGraph(FrameGraph::NodeId presentDepNode, FrameGraph::QueueId depQueueId);

    FrameGraph& getFrameGraph() { return frameGraph; }
    const FrameGraph& getFrameGraph() const { return frameGraph; }

    virtual void simulate(FrameId frameId) = 0;
    virtual void beforeDraw(FrameId frameId) = 0;
    virtual AcquiredImageData record(FrameId frameId) = 0;
    virtual void readback(FrameId frameId) = 0;
    virtual void releaseSwapchainResources() = 0;
    virtual void createSwapchainResources(const drv::Swapchain& swapchain) = 0;

    drv::TimelineSemaphorePool* getSemaphorePool() { return &semaphorePool; }

 private:
    friend class AccessValidationCallback;

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

    EngineConfig config;
    Args launchArgs;

    Logger logger;
    ErrorCallback callback;
    std::unique_ptr<CoreContext> coreContext;
    GarbageSystem garbageSystem;
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
    drv::TimelineSemaphorePool semaphorePool;
    drv::Swapchain swapchain;
    EventPool eventPool;
    SyncBlock syncBlock;
    // ResourceManager resourceMgr;
    // EntityManager entityManager;
    drv::ResourceLocker resourceLocker;
    FrameGraph frameGraph;
    RuntimeStats runtimeStats;

    FrameGraph::NodeId inputSampleNode;
    FrameGraph::NodeId presentFrameNode;
    QueueInfo queueInfos;

    uint32_t acquireImageSemaphoreId = 0;
    FrameId firstPresentableFrame = 0;
    enum class SwapchainState
    {
        UNKNOWN,
        OK,
        OKAY,
        INVALID
    };
    std::atomic<SwapchainState> swapchainState = SwapchainState::UNKNOWN;

    mutable std::mutex executionMutex;
    mutable std::mutex swapchainMutex;
    mutable std::mutex mainKernelMutex;
    mutable std::mutex swapchainRecreationMutex;
    std::condition_variable mainKernelCv;
    std::condition_variable mainKernelSwapchainCv;
    std::condition_variable beforeDrawSwapchainCv;
    std::atomic<bool> swapchainRecreationRequired = {false};
    std::atomic<bool> swapchainRecreationPossible = {false};

    void simulationLoop();
    void beforeDrawLoop();
    void recordCommandsLoop();
    void executeCommandsLoop();
    void readbackLoop();
    void mainLoopKernel();
    bool execute(ExecutionPackage&& package);
    void present(drv::SwapchainPtr swapchain, FrameId frame, uint32_t imageIndex,
                 uint32_t semaphoreIndex);
    bool sampleInput(FrameId frameId);

    static drv::PhysicalDevice::SelectionInfo get_device_selection_info(
      drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions,
      const ShaderBin& shaderBin);
    static drv::Swapchain::CreateInfo get_swapchain_create_info(const EngineConfig& config,
                                                                drv::QueuePtr present_queue,
                                                                drv::QueuePtr render_queue);
    drv::Swapchain::OldSwapchinData recreateSwapchain();
};
