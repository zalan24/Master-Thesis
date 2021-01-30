#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <drv.h>
#include <drv_queue_manager.h>
#include <drv_wrappers.h>
#include <drvcmdbufferbank.h>
#include <drvlane.h>

#include <entitymanager.h>
#include <input.h>
#include <inputmanager.h>
// #include <renderer.h>
#include <corecontext.h>
#include <resourcemanager.h>
#include <serializable.h>

class ExecutionQueue;

class Engine
{
 public:
    using FrameId = size_t;

    struct Config final : public ISerializable
    {
        int screenWidth;
        int screenHeight;
        int imagesInSwapchain;
        int maxFramesInExecutionQueue;
        // TODO wait on this
        int maxFramesOnGPU;
        std::string title;
        std::string driver;
        int inputBufferSize;
        int stackMemorySizeKb;
        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;
    };

    Engine(const Config& config, ResourceManager::ResourceInfos resource_infos);
    Engine(const std::string& configFile, ResourceManager::ResourceInfos resource_infos);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void gameLoop();

    EntityManager* getEntityManager() { return &entityManager; }
    const EntityManager* getEntityManager() const { return &entityManager; }

    // Renderer* getRenderer() { return &renderer; }
    // const Renderer* getRenderer() const { return &renderer; }

    // ResourceManager* getResMgr() { return &resourceMgr; }
    // const ResourceManager* getResMgr() const { return &resourceMgr; }

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
        drv::Semaphore imageAvailableSemaphore;
        SyncBlock(drv::LogicalDevicePtr device);
    };

    using FrameId = size_t;

    Config config;

    ErrorCallback callback;
    drv::Input input;
    drv::InputManager inputManager;
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
    SyncBlock syncBlock;
    ResourceManager resourceMgr;
    EntityManager entityManager;
    // Renderer renderer;

    struct RenderState
    {
        std::atomic<bool> quit = false;
        std::atomic<bool> canSimulate = true;
        std::atomic<bool> canRecord = false;
        // std::atomic<bool> canExecute = false;
        std::atomic<FrameId> simulationFrame = 0;
        std::atomic<FrameId> recordFrame = 0;
        std::atomic<FrameId> executeFrame = 0;
        std::atomic<FrameId> recreateSwapchain = 0;
        std::atomic<FrameId> swapchainCreated = 0;
        std::mutex simulationMutex;
        std::mutex recordMutex;
        // std::mutex executeMutex;
        std::condition_variable simulationCV;
        std::condition_variable recordCV;
        ExecutionQueue* executionQueue;
        // std::condition_variable executeCV;
    };

    void simulationLoop(RenderState* state);
    void recordCommandsLoop(RenderState* state);
    void executeCommandsLoop(RenderState* state);
    void present(RenderState* state, FrameId presentFrame);
    void sampleInput();

    static drv::PhysicalDevice::SelectionInfo get_device_selection_info(
      drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions);
    static drv::Swapchain::CreateInfo get_swapchain_create_info(const Config& config);
};
