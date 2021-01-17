#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <entitymanager.h>
#include <renderer.h>
#include <resourcemanager.h>
#include <serializable.h>
#include <window.h>

class Engine
{
 public:
    struct Config final : public ISerializable
    {
        int screenWidth;
        int screenHeight;
        std::string title;
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

    Renderer* getRenderer() { return &renderer; }
    const Renderer* getRenderer() const { return &renderer; }

    ResourceManager* getResMgr() { return &resourceMgr; }
    const ResourceManager* getResMgr() const { return &resourceMgr; }

 private:
    struct GlLoader
    {
        GlLoader();
        ~GlLoader();
        GlLoader(const GlLoader&) = delete;
        GlLoader& operator=(const GlLoader&) = delete;
    };

    using FrameId = size_t;

    Config config;

    Window window;
    Renderer renderer;
    ResourceManager resourceMgr;
    EntityManager entityManager;

    FrameId simulationFrame = 0;
    FrameId renderFrame = 0;

    std::mutex mutex;
    std::condition_variable renderCV;
    std::condition_variable simulationCV;

    enum LoopState
    {
        SIMULATE,
        RENDER,
        SIMULATION_END
    };

    void simulationLoop(bool* quit, LoopState* state);
};
