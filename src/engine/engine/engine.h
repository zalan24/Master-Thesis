#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
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

#include <physics.h>
#include <renderpass.h>
#include <runtimestats.h>
#include <shaderbin.h>
#include <cmdBuffer.hpp>
#include <cmdRecorder.hpp>
#include <oneTimeCmdBuffer.hpp>

#include "bufferstager.h"
#include "imagestager.h"
#include "resources.hpp"
#include "timestampcmdbuffers.h"

struct ExecutionPackage;

struct EngineConfig final : public IAutoSerializable<EngineConfig>
{
    REFLECTABLE((uint32_t)screenWidth, (uint32_t)screenHeight, (uint32_t)imagesInSwapchain,
                (uint32_t)maxFramesInExecutionQueue, (uint32_t)maxFramesInFlight,
                (uint32_t)slopHistorySize, (std::string)title, (std::string)driver,
                (uint32_t)inputBufferSize, (uint32_t)stackMemorySizeKb, (uint32_t)frameMemorySizeKb,
                (std::string)logs)
};

struct EngineOptions final : public IAutoSerializable<EngineOptions>
{
    enum RefreshRateMode
    {
        UNLIMITED = 0,
        LIMITED = 1,
        DISCRETIZED = 2
    };

    static std::string get_enum_name(RefreshRateMode mode) {
        switch (mode) {
            case UNLIMITED:
                return "unlimited";
            case DISCRETIZED:
                return "discretized";
            case LIMITED:
                return "limited";
        }
    }

    static int32_t get_enum(const std::string& s) {
        for (const RefreshRateMode& m : {UNLIMITED, DISCRETIZED, LIMITED})
            if (get_enum_name(m) == s)
                return static_cast<int32_t>(m);
        throw std::runtime_error("Couldn't decode enum");
    }

    REFLECTABLE((bool)latencyReduction, (float)desiredSlop, (bool)perfMetrics_window,
                (bool)perfMetrics_fps, (bool)perfMetrics_theoreticalFps, (bool)perfMetrics_cpuWork,
                (bool)perfMetrics_execWork, (bool)perfMetrics_deviceWork, (bool)perfMetrics_latency,
                (bool)perfMetrics_slop, (bool)perfMetrics_perFrameSlop, (bool)perfMetrics_sleep,
                (bool)perfMetrics_execDelay, (bool)perfMetrics_deviceDelay, (bool)perfMetrics_work,
                (bool)perfMetrics_skippedDelayed, (bool)manualLatencyReduction,
                (float)manualSleepTime, (float)targetRefreshRate, (RefreshRateMode)refreshMode,
                (float)workTimeSmoothing)

    EngineOptions()
      : latencyReduction(false),
        desiredSlop(4),
        // workPrediction(0),
        perfMetrics_window(true),
        perfMetrics_fps(true),
        perfMetrics_theoreticalFps(true),
        perfMetrics_cpuWork(true),
        perfMetrics_execWork(true),
        perfMetrics_deviceWork(true),
        perfMetrics_latency(true),
        perfMetrics_slop(true),
        perfMetrics_perFrameSlop(true),
        perfMetrics_sleep(true),
        perfMetrics_execDelay(true),
        perfMetrics_deviceDelay(true),
        perfMetrics_work(true),
        perfMetrics_skippedDelayed(true),
        manualLatencyReduction(false),
        manualSleepTime(0.0f),
        targetRefreshRate(60.0f),
        refreshMode(UNLIMITED),
        workTimeSmoothing(0.5f) {}
};

struct TransformRecordEntry final : public IAutoSerializable<TransformRecordEntry>
{
    REFLECTABLE((glm::quat)orientation, (glm::vec3)position, (float)timeMs)

    TransformRecordEntry() = default;
    TransformRecordEntry(const glm::quat& _orientation, const glm::vec3& _position, float _timeMs)
      : orientation(_orientation), position(_position), timeMs(_timeMs) {}
};

struct TransformRecord final : public IAutoSerializable<TransformRecord>
{
    REFLECTABLE((std::vector<TransformRecordEntry>)entries)
};

struct PerformanceCaptureCpuPackage final : public IAutoSerializable<PerformanceCaptureCpuPackage>
{
    REFLECTABLE((std::string)name, (uint64_t)frameId, (uint32_t)packageId, (double)slopDuration,
                (double)availableTime, (double)resAvailableTime, (double)startTime, (double)endTime,
                (std::set<uint32_t>)depended, (std::set<uint32_t>)dependent,
                (std::set<uint32_t>)execDepended, (std::set<uint32_t>)deviceDepended,
                (std::map<std::string, uint64_t>)gpuDoneDep)
};

struct PerformanceCaptureInterval final : public IAutoSerializable<PerformanceCaptureInterval>
{
    REFLECTABLE((double)startTime, (double)endTime)
};

struct PerformanceCaptureExecutionPackage final
  : public IAutoSerializable<PerformanceCaptureExecutionPackage>
{
    REFLECTABLE(
      // (std::string) name,
      (uint32_t)packageId, (uint32_t)sourcePackageId, (double)slopDuration, (double)issueTime,
      (double)startTime, (double)endTime, (bool)minimalDelayInFrame)
};

struct PerformanceCaptureDevicePackage final
  : public IAutoSerializable<PerformanceCaptureDevicePackage>
{
    REFLECTABLE(
      // (std::string) name,
      (uint32_t)sourceExecPackageId, (double)slopDuration, (double)submissionTime,
      (double)startTime, (double)endTime)
};

struct PerformanceCaptureData final : public IAutoSerializable<PerformanceCaptureData>
{
    REFLECTABLE(
      (uint64_t)frameId, (double)fps, (double)frameTime, (double)softwareLatency,
      (double)latencySlop, (double)sleepTime, (double)executionDelay, (double)deviceDelay,
      (double)workTime, (double)frameEndFixPoint,
      (std::map<std::string, std::map<std::string, std::vector<PerformanceCaptureCpuPackage>>>)
        stageToThreadToPackageList,
      (std::vector<std::string>)cpuStageOrder,
      (std::map<uint32_t, PerformanceCaptureInterval>)executionIntervals,
      (std::vector<PerformanceCaptureExecutionPackage>)executionPackages,
      (std::map<std::string, std::vector<PerformanceCaptureDevicePackage>>)queueToDevicePackageList,
      (EngineOptions)engineOptions)
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
    bool popToggleFreeCame() { return std::exchange(toggleFreeCame, false); }
    bool popToggleRecording() { return std::exchange(toggleRecording, false); }
    bool isPhysicsFrozen() const { return physicsFrozen; }

 protected:
    bool processKeyboard(const Input::KeyboardEvent&) override;
    bool processMouseButton(const Input::MouseButtenEvent&) override;
    bool processMouseMove(const Input::MouseMoveEvent&) override;
    // bool processScroll(const Input::ScrollEvent&) override;

 private:
    bool perfCapture = false;
    bool clicking = false;
    bool toggleFreeCame = false;
    bool physicsFrozen = false;
    bool toggleRecording = false;
    double mX;
    double mY;
};

struct CameraControlInfo
{
    glm::vec3 translation = glm::vec3(0, 0, 0);
    glm::vec2 rotation;
};

class FreeCamInput final : public InputListener
{
 public:
    FreeCamInput() : InputListener(true) {
        boost = hasPrevData = left = right = forward = backward = up = 0;
    }
    ~FreeCamInput() override {}

    CursorMode getCursorMode() override final { return LOCK; }

    CameraControlInfo popCameraControls();

 protected:
    bool processKeyboard(const Input::KeyboardEvent& event) override final;
    bool processMouseMove(const Input::MouseMoveEvent& event) override final;

 private:
    FrameGraph::Clock::time_point lastSample;
    uint16_t hasPrevData : 1;
    uint16_t left : 1;
    uint16_t right : 1;
    uint16_t forward : 1;
    uint16_t backward : 1;
    uint16_t up : 1;
    uint16_t boost : 1;

    glm::vec3 speed = glm::vec3(0, 0, 0);
    glm::vec2 drag = glm::vec2(10, 50);
    float normalSpeed = 2;
    float fastSpeed = 50;
    float rotationSpeed = 3;
    glm::vec2 cameraRotate = glm::vec2(0, 0);
};

template <uint32_t N>
struct StatCalculator
{
 public:
    void feed(double value) {
        std::unique_lock<std::mutex> lock(mutex);
        values[(count++) % N] = value;
        sum += value;
        if ((count % N) == 0) {
            avg = sum / N;
            sum = 0;
            stdDiv = 0;
            for (uint32_t i = 0; i < N; ++i)
                stdDiv += (values[i] - avg) * (values[i] - avg);
            stdDiv = sqrt(stdDiv / N);
        }
    }
    bool hasInfo() const { return count >= N; }
    double getAvg() const {
        std::unique_lock<std::mutex> lock(mutex);
        return avg;
    }
    double getStdDiv() const {
        std::unique_lock<std::mutex> lock(mutex);
        return stdDiv;
    }

 private:
    uint32_t count = 0;
    std::array<double, N> values;

    mutable std::mutex mutex;

    double sum = 0;
    double avg = 0;
    double stdDiv = 0;
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
        QueueId id;
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
    //                                         FrameId frameId, QueueId queueId);

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

    bool isFrozen() const { return mouseListener.isPhysicsFrozen(); }

    FrameGraph::Clock::time_point getStartupTime() const {return frameEndFixPoint;}

 protected:
    // Needs to be called from game implementation after finishing the framegraph
    void buildFrameGraph();
    void initImGui(drv::RenderPass* imGuiRenderpass);

    FrameGraph& getFrameGraph() { return frameGraph; }
    const FrameGraph& getFrameGraph() const { return frameGraph; }

    virtual void simulate(FrameId frameId) = 0;
    virtual void beforeDraw(FrameId frameId) = 0;
    virtual void record(const AcquiredImageData& swapchainData, EngineCmdBufferRecorder* recorder,
                        FrameId frameId) = 0;
    virtual void recordGameUI(FrameId /*frameId*/) {}
    virtual void recordMenuOptionsUI(FrameId /*frameId*/) {}

    virtual void lockResources(TemporalResourceLockerDescriptor& resourceDesc, FrameId frameId) = 0;
    virtual void readback(FrameId frameId) = 0;
    virtual void releaseSwapchainResources() = 0;
    virtual void createSwapchainResources(const drv::Swapchain& swapchain) = 0;

    void recordImGui(const AcquiredImageData& swapchainData, EngineCmdBufferRecorder* recorder,
                     FrameId frameId);

    drv::TimelineSemaphorePool* getSemaphorePool() { return &semaphorePool; }

    void transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                            FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                            ImageStager::StagerId stagerId);
    void transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                            FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                            ImageStager::StagerId stagerId, uint32_t layer, uint32_t mip);
    void transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                            FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                            ImageStager::StagerId stagerId,
                            const drv::ImageSubresourceRange& subres);
    void transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                          FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                          ImageStager::StagerId stagerId);
    void transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                          FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                          ImageStager::StagerId stagerId, uint32_t layer, uint32_t mip);
    void transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                          FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                          ImageStager::StagerId stagerId, const drv::ImageSubresourceRange& subres);

    void initPhysicsEntitySystem();
    void initRenderEntitySystem();
    void initCursorEntitySystem();
    void initBeforeDrawEntitySystem();

    struct EntityRenderData
    {
        glm::mat4 modelTm;
        glm::vec3 albedo;
        std::string shape;
    };

    struct RendererData
    {
        glm::vec3 sunDir;
        glm::vec3 sunLight;
        glm::vec3 ambientLight;
        glm::vec3 eyePos;
        glm::vec3 eyeDir;
        bool latencyFlash;
        bool cameraRecord;
        glm::vec2 cursorPos;
        float ratio;
    };

    uint32_t getNumEntitiesToRender() const { return uint32_t(entitiesToDraw.size()); }
    const EntityRenderData* getEntitiesToRender() const { return entitiesToDraw.data(); }
    const RendererData& getRenderData(FrameId frame) const {
        return perFrameTempInfo[frame % perFrameTempInfo.size()].renderData;
    }

    NodeId getMainRecordNode() const { return mainRecordNode; }

    void createPerformanceCapture(FrameId targetFrame);

    bool isInFreecam() const { return inFreeCam; }

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

    struct ImGuiIniter
    {
        ImGuiIniter(IWindow* window, drv::InstancePtr instance,
                    drv::PhysicalDevicePtr physicalDevice, drv::LogicalDevicePtr device,
                    drv::QueuePtr renderQueue, drv::QueuePtr transferQueue,
                    drv::RenderPass* renderpass, uint32_t minSwapchainImages,
                    uint32_t swapchainImages);
        ~ImGuiIniter();

        ImGuiIniter(const ImGuiIniter&) = delete;
        ImGuiIniter& operator=(const ImGuiIniter&) = delete;

        IWindow* window;
    };

    std::default_random_engine generator;
    mutable std::mutex generatorMutex;

    float genFloat01() {
        std::uniform_real_distribution<float> dist01(0, 1);
        std::unique_lock<std::mutex> randomLock(generatorMutex);
        return dist01(generator);
    }

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
    bool inFreeCam = false;
    std::unique_ptr<FreeCamInput> freecamListener;
    std::unique_ptr<InputListener> imGuiInputListener;
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
    Physics physics;
    EntityManager entityManager;
    std::unique_ptr<ImGuiIniter> imGuiIniter;
    EngineOptions engineOptions;

    NodeId inputSampleNode;
    NodeId physicsSimulationNode;
    NodeId mainRecordNode;
    NodeId acquireSwapchainNode;
    NodeId presentFrameNode;
    QueueInfo queueInfos;
    EntityManager::EntitySystemInfo physicsEntitySystem;
    EntityManager::EntitySystemInfo emitterEntitySystem;
    EntityManager::EntitySystemInfo renderEntitySystem;
    EntityManager::EntitySystemInfo cameraEntitySystem;
    EntityManager::EntitySystemInfo benchmarkEntitySystem;
    drv::Clock::time_point nextTimelineCalibration;
    drv::Clock::time_point lastLatencyFlashClick;

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
    mutable std::mutex inputWaitMutex;
    mutable std::mutex inputSamplingMutex;
    std::condition_variable mainKernelCv;
    std::condition_variable waitForInputCv;
    std::condition_variable mainKernelSwapchainCv;
    std::condition_variable beforeDrawSwapchainCv;
    std::atomic<bool> swapchainRecreationRequired = {false};
    std::atomic<bool> swapchainRecreationPossible = {false};
    std::filesystem::file_time_type workLoadFileModificationDate;
    bool wantToQuit = false;
    bool latencyOptionsOpen = false;
    mutable std::mutex latencyInfoMutex;
    FrameGraphSlops::ExtendedLatencyInfo latestLatencyInfo;
    FrameGraphSlops::ExtendedLatencyInfo captureLatencyInfo;
    FrameGraph::Clock::time_point frameEndFixPoint;
    FrameGraph::Clock::time_point startupTime;
    res::ImageSet captureImage;
    ImageStager captureImageStager;
    struct FrameHistoryInfo
    {
        std::chrono::nanoseconds duration;
        FrameGraph::Clock::time_point estimatedEnd;
        std::chrono::nanoseconds headRoom;
    };
    std::vector<FrameHistoryInfo> frameHistory;
    double worTimeAvgMs = 0;
    double frameOffsetAvgMs = 0;
    // double refreshTimeCpuAvgMs = 0;
    // double refreshTimeExecAvgMs = 0;
    // double refreshTimeDeviceAvgMs = 0;
    int64_t completedFrameIntervalId = 0;

    std::vector<EntityRenderData> entitiesToDraw;
    FrameId perfCaptureFrame = INVALID_FRAME;

    struct PerFrameTempInfo
    {
        drv::ImagePtr captureImage = drv::get_null_ptr<drv::ImagePtr>();
        bool captureHappening = false;
        RendererData renderData;
        CameraControlInfo cameraControls;
        FrameGraph::Clock::time_point frameStartTime;
        double deltaTimeSec = 0;
        float benchmarkPeriod = 0;
    };
    std::vector<PerFrameTempInfo> perFrameTempInfo;

    struct BenchmarkData
    {
        // time measured in ms
        float period;
        float fps;
        float latency;
        float latencySlop;
        float cpuWork;
        float execWork;
        float deviceWork;
        float workTime;
        float missRate;  // smoothened
    };
    std::deque<BenchmarkData> benchmarkData;

    StatCalculator<32> fpsStats;
    StatCalculator<32> theoreticalFpsStats;
    StatCalculator<32> cpuWorkStats;
    StatCalculator<32> cpuOffsetStats;
    StatCalculator<32> execWorkStats;
    StatCalculator<32> execOffsetStats;
    StatCalculator<32> deviceWorkStats;
    StatCalculator<32> deviceOffsetStats;
    StatCalculator<32> latencyStats;
    StatCalculator<32> slopStats;
    StatCalculator<32> perFrameSlopStats;
    StatCalculator<32> waitTimeStats;
    StatCalculator<32> execDelayStats;
    StatCalculator<32> deviceDelayStats;
    StatCalculator<32> workStats;
    StatCalculator<8> skippedDelayed;

    struct SubmissionTimestampsInfo
    {
        drv::Clock::time_point submissionTime;
        NodeId node;
        drv::CmdBufferId submission;
        drv::QueuePtr queue;
        QueueId queueId;
        uint32_t beginTimestampBufferIndex;
        uint32_t endTimestampBufferIndex;
    };
    std::vector<std::vector<SubmissionTimestampsInfo>> timestsampRingBuffer;

    bool isRecording = false;
    bool hasStartedRecording = false;
    FrameGraph::Clock::time_point cameraRecordStart;
    TransformRecord cameraRecord;

    void simulationLoop();
    void beforeDrawLoop();
    void recordCommandsLoop();
    void executeCommandsLoop();
    void readbackLoop(volatile bool* finished);
    void mainLoopKernel();
    bool execute(ExecutionPackage&& package);
    void present(drv::SwapchainPtr swapchain, FrameId frame, uint32_t imageIndex,
                 uint32_t semaphoreIndex);
    bool sampleInput(FrameId frameId);
    bool simulatePhysics(FrameId frameId);
    void drawUI(FrameId frameId);
    PerformanceCaptureData generatePerfCapture(
      FrameId lastReadyFrame, const FrameGraphSlops::ExtendedLatencyInfo& latency) const;
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
                          const EntityManager::EntitySystemParams& params, Entity* entity,
                          Entity::EntityId id, FlexibleArray<Entity, 4>& outEntities);
    static void esBeforeDraw(EntityManager* entityManager, Engine* engine,
                             FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                             const EntityManager::EntitySystemParams& params, Entity* entity,
                             Entity::EntityId id, FlexibleArray<Entity, 4>& outEntities);
    static void esCamera(EntityManager* entityManager, Engine* engine,
                         FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                         const EntityManager::EntitySystemParams& params, Entity* entity,
                         Entity::EntityId id, FlexibleArray<Entity, 4>& outEntities);
    static void esBenchmark(EntityManager* entityManager, Engine* engine,
                            FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                            const EntityManager::EntitySystemParams& params, Entity* entity,
                            Entity::EntityId id, FlexibleArray<Entity, 4>& outEntities);
    static void esEmitter(EntityManager* entityManager, Engine* engine,
                          FrameGraph::NodeHandle* nodeHandle, FrameGraph::Stage stage,
                          const EntityManager::EntitySystemParams& params, Entity* entity,
                          Entity::EntityId id, FlexibleArray<Entity, 4>& outEntities);
};
