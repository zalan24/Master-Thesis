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
#include <drvrenderpass.h>
#include <drvresourcelocker.h>
#include <drvsemaphorepool.h>

#include <entitymanager.h>
#include <eventpool.h>
#include <framegraph.h>
#include <timestamppool.h>

#include <corecontext.h>
#include <garbagesystem.h>
#include <input.h>
#include <inputmanager.h>
#include <serializable.h>

#include <runtimestats.h>
#include <shaderbin.h>
#include <cmdBuffer.hpp>
#include <oneTimeCmdBuffer.hpp>

#include "bufferstager.h"
#include "imagestager.h"
#include "resources.hpp"
#include "timestampcmdbuffers.h"

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

struct PerformanceCaptureCpuPackage final : public IAutoSerializable<PerformanceCaptureCpuPackage>
{
    REFLECTABLE
    (
        (std::string) name,
        (uint64_t) frameId,
        (uint32_t) packageId,
        (double) availableTime,
        (double) resAvailableTime,
        (double) startTime,
        (double) endTime,
        (std::set<uint32_t>) depended,
        (std::set<uint32_t>) dependent
    )
};

struct PerformanceCaptureInterval final : public IAutoSerializable<PerformanceCaptureInterval>
{
    REFLECTABLE
    (
        (double) startTime,
        (double) endTime
    )
};

struct PerformanceCaptureExecutionPackage final : public IAutoSerializable<PerformanceCaptureExecutionPackage>
{
    REFLECTABLE
    (
        (std::string) name,
        (uint32_t) sourcePackageId,
        (double) issueTime,
        (double) startTime,
        (double) endTime,
        (bool) minimalDelayInFrame
    )
};

struct PerformanceCaptureData final : public IAutoSerializable<PerformanceCaptureData>
{
    REFLECTABLE
    (
        (uint64_t) frameId,
        (double) fps,
        (double) frameTime,
        (double) executionDelay,
        (double) deviceDelay,
        (std::map<std::string, std::map<std::string, std::vector<PerformanceCaptureCpuPackage>>>) stageToThreadToPackageList,
        (std::map<uint32_t, PerformanceCaptureInterval>) executionIntervals,
        (std::vector<PerformanceCaptureExecutionPackage>) executionPackages
    )
};

class EngineInputListener final : public InputListener
{
 public:
    EngineInputListener() : InputListener(false) {}
    // EngineInputListener(Renderer* _renderer) : InputListener(false), renderer(_renderer) {}
    ~EngineInputListener() override {}

    CursorMode getCursorMode() override final { return DONT_CARE; }

    bool isClicking() const { return clicking; }
    glm::vec2 getMousePos() const { return {mX, mY}; }
    bool popNeedPerfCapture() { return std::exchange(perfCapture, false); }

 protected:
    bool processKeyboard(const Input::KeyboardEvent&) override;
    bool processMouseButton(const Input::MouseButtenEvent&) override;
    bool processMouseMove(const Input::MouseMoveEvent&) override;
    // bool processScroll(const Input::ScrollEvent&) override;

 private:
    bool perfCapture = false;
    bool clicking = false;
    double mX;
    double mY;
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
        std::string sceneToLoad;
        bool clearRuntimeStats = false;
    };

    struct Resources
    {
        // folders
        std::string assets;
        std::string textures;
    };

    Engine(int argc, char* argv[], const EngineConfig& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           const Resources& resources, Args args);
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
    void buildFrameGraph();

    FrameGraph& getFrameGraph() { return frameGraph; }
    const FrameGraph& getFrameGraph() const { return frameGraph; }

    virtual void simulate(FrameId frameId) = 0;
    virtual void beforeDraw(FrameId frameId) = 0;
    virtual void record(const AcquiredImageData& swapchainData, drv::DrvCmdBufferRecorder* recorder,
                        FrameId frameId) = 0;
    virtual void lockResources(TemporalResourceLockerDescriptor& resourceDesc, FrameId frameId) = 0;
    virtual void readback(FrameId frameId) = 0;
    virtual void releaseSwapchainResources() = 0;
    virtual void createSwapchainResources(const drv::Swapchain& swapchain) = 0;

    drv::TimelineSemaphorePool* getSemaphorePool() { return &semaphorePool; }

    void transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, FrameGraph::QueueId queue, FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                            ImageStager::StagerId stagerId);
    void transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, FrameGraph::QueueId queue, FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                            ImageStager::StagerId stagerId, uint32_t layer, uint32_t mip);
    void transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, FrameGraph::QueueId queue, FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                            ImageStager::StagerId stagerId, const drv::ImageSubresourceRange& subres);
    void transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, FrameGraph::QueueId queue, FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                          ImageStager::StagerId stagerId);
    void transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, FrameGraph::QueueId queue, FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                          ImageStager::StagerId stagerId, uint32_t layer, uint32_t mip);
    void transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, FrameGraph::QueueId queue, FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                          ImageStager::StagerId stagerId, const drv::ImageSubresourceRange& subres);

    void initPhysicsEntitySystem();
    void initRenderEntitySystem();
    void initCursorEntitySystem();
    void initBeforeDrawEntitySystem();

    void drawEntities(drv::DrvCmdBufferRecorder* recorder, drv::ImagePtr targetImage);

    NodeId getMainRecordNode() const { return mainRecordNode; }

    void createPerformanceCapture(FrameId targetFrame);

 private:
    static constexpr uint64_t firstTimelineCalibrationTimeMs = 1000;
    static constexpr uint64_t otherTimelineCalibrationTimeMs = 10000;
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

    struct EntityRenderData
    {
        glm::vec2 relBottomLeft;
        glm::vec2 relTopRight;
        uint32_t textureId;
        float z;
    };

    EngineConfig config;
    Resources resourceFolders;
    Args launchArgs;
    std::string workLoadFile;

    Logger logger;
    ErrorCallback callback;
    std::unique_ptr<CoreContext> coreContext;
    GarbageSystem garbageSystem;
    ShaderBin shaderBin;
    Input input;
    InputManager inputManager;
    EngineInputListener mouseListener;
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
    TimestampPool timestampPool;
    DynamicTimestampCmdBufferPool timestampCmdBuffers;
    drv::Swapchain swapchain;
    EventPool eventPool;
    SyncBlock syncBlock;
    // ResourceManager resourceMgr;
    drv::ResourceLocker resourceLocker;
    FrameGraph frameGraph;
    RuntimeStats runtimeStats;
    EntityManager entityManager;

    NodeId inputSampleNode;
    NodeId mainRecordNode;
    NodeId acquireSwapchainNode;
    NodeId presentFrameNode;
    QueueInfo queueInfos;
    EntityManager::EntitySystemInfo physicsEntitySystem;
    EntityManager::EntitySystemInfo renderEntitySystem;
    EntityManager::EntitySystemInfo cursorEntitySystem;
    EntityManager::EntitySystemInfo latencyFlashEntitySystem;
    EntityManager::EntitySystemInfo cameraEntitySystem;
    drv::Clock::time_point nextTimelineCalibration;

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
    std::filesystem::file_time_type workLoadFileModificationDate;

    std::vector<EntityRenderData> entitiesToDraw;
    FrameId perfCaptureFrame = INVALID_FRAME;

    struct SubmissionTimestampsInfo
    {
        drv::Clock::time_point submissionTime;
        NodeId node;
        drv::CmdBufferId submission;
        drv::QueuePtr queue;
        uint32_t beginTimestampBufferIndex;
        uint32_t endTimestampBufferIndex;
    };
    std::vector<std::vector<SubmissionTimestampsInfo>> timestsampRingBuffer;

    void simulationLoop();
    void beforeDrawLoop();
    void recordCommandsLoop();
    void executeCommandsLoop();
    void readbackLoop(volatile bool *finished);
    void mainLoopKernel();
    bool execute(ExecutionPackage&& package);
    void present(drv::SwapchainPtr swapchain, FrameId frame, uint32_t imageIndex,
                 uint32_t semaphoreIndex);
    bool sampleInput(FrameId frameId);
    PerformanceCaptureData generatePerfCapture(FrameId lastReadyFrame) const;
    AcquiredImageData mainRecord(FrameId frameId);

    static drv::PhysicalDevice::SelectionInfo get_device_selection_info(
      drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions,
      const ShaderBin& shaderBin);
    static drv::Swapchain::CreateInfo get_swapchain_create_info(const EngineConfig& config,
                                                                drv::QueuePtr present_queue,
                                                                drv::QueuePtr render_queue);
    drv::Swapchain::OldSwapchinData recreateSwapchain();

    static void esPhysics(EntityManager* entityManager, Engine* engine,
                          FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                          const EntityManager::EntitySystemParams& params, Entity* entity);
    static void esBeforeDraw(EntityManager* entityManager, Engine* engine,
                             FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                             const EntityManager::EntitySystemParams& params, Entity* entity);
    static void esCursor(EntityManager* entityManager, Engine* engine,
                         FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                         const EntityManager::EntitySystemParams& params, Entity* entity);
    static void esLatencyFlash(EntityManager* entityManager, Engine* engine,
                               FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                               const EntityManager::EntitySystemParams& params, Entity* entity);
    static void esCamera(EntityManager* entityManager, Engine* engine,
                               FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                               const EntityManager::EntitySystemParams& params, Entity* entity);
};
