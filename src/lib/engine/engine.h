#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <drv.h>
#include <drv_queue_manager.h>
#include <drv_wrappers.h>

#include <entitymanager.h>
// #include <renderer.h>
// #include <resourcemanager.h>
#include <serializable.h>
// #include <window.h>

class Engine
{
 public:
    using FrameId = size_t;

    struct Config final : public ISerializable
    {
        int screenWidth;
        int screenHeight;
        std::string title;
        std::string driver;
        void gatherEntries(std::vector<ISerializable::Entry>& entries) const override;
    };

    Engine(const Config& config);
    Engine(const std::string& configFile);
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
    struct GlLoader
    {
        GlLoader();
        ~GlLoader();
        GlLoader(const GlLoader&) = delete;
        GlLoader& operator=(const GlLoader&) = delete;
    };

    struct DriverSelector
    {
        DriverSelector(drv::Driver driver);
    };

    using FrameId = size_t;

    Config config;

    DriverSelector driverSelector;
    drv::DriverWrapper driver;
    drv::Instance drvInstance;
    drv::PhysicalDevice physicalDevice;
    drv::LogicalDevice device;
    drv::QueueManager queueManager;
    // ResourceManager resourceMgr;
    EntityManager entityManager;
    // Window window;
    // Renderer renderer;

    struct RenderState
    {
        std::atomic<bool> quit = false;
        std::atomic<bool> canSimulate = true;
        std::atomic<bool> canRecord = false;
        // std::atomic<bool> canExecute = false;
        std::atomic<FrameId> simulationFrame = 0;
        std::atomic<FrameId> recordFrame = 0;
        // std::atomic<FrameId> executeFrame = 0;
        std::mutex simulationMutex;
        std::mutex recordMutex;
        // std::mutex executeMutex;
        std::condition_variable simulationCV;
        std::condition_variable recordCV;
        // std::condition_variable executeCV;
    };

    void simulationLoop(RenderState* state);
    void recordCommandsLoop(RenderState* state);
    void executeCommandsLoop(RenderState* state);
    static drv::PhysicalDevice::SelectionInfo get_device_selection_info(drv::InstancePtr instance);
};
