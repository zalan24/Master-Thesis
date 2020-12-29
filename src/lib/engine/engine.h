#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <entitymanager.h>
#include <renderer.h>
#include <serializable.h>

class Engine
{
 public:
    struct Config : public ISerializable
    {
        int screenWidth;
        int screenHeight;
        void gatherEntries(std::vector<ISerializable::Entry>& entries) const override;
    };

    Engine(const Config& config);
    Engine(const std::string& configFile);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void gameLoop();

 private:
    using FrameId = size_t;

    Config config;

    EntityManager entityManager;
    // Renderer renderer; // TODO

    FrameId simulationFrame = 0;
    FrameId renderFrame = 0;

    std::thread simulationThread;
    std::mutex mutex;
};
