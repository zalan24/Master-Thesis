#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <entitymanager.h>
#include <renderer.h>
#include <serializable.h>
#include <window.h>

class Engine
{
 public:
    struct Config : public ISerializable
    {
        int screenWidth;
        int screenHeight;
        std::string title;
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

    Renderer* getRenderer() { return &renderer; }
    const Renderer* getRenderer() const { return &renderer; }

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

    EntityManager entityManager;
    Window window;
    Renderer renderer;

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
