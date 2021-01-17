#include "engine.h"

#include <fstream>

#include <rendercontext.h>

void Engine::Config::writeJson(json& out) const {
    WRITE_OBJECT(screenWidth, out);
    WRITE_OBJECT(screenHeight, out);
    WRITE_OBJECT(title, out);
}

void Engine::Config::readJson(const json& in) {
    READ_OBJECT(screenWidth, in);
    READ_OBJECT(screenHeight, in);
    READ_OBJECT(title, in);
}

static Engine::Config get_config(const std::string& file) {
    std::ifstream in(file);
    assert(in.is_open());
    Engine::Config config;
    config.read(in);
    return config;
}

Engine::Engine(const std::string& configFile, ResourceManager::ResourceInfos resource_infos)
  : Engine(get_config(configFile), std::move(resource_infos)) {
}

Engine::Engine(const Config& cfg, ResourceManager::ResourceInfos resource_infos)
  : config(cfg),
    resourceMgr(std::move(resource_infos)),
    window(config.screenWidth, config.screenHeight, config.title) {
}

Engine::~Engine() {
    // TODO
    checkError();
}

void Engine::simulationLoop(bool* quit, LoopState* state) {
    simulationFrame = 0;
    while (!*quit) {
        {
            std::unique_lock<std::mutex> lk(mutex);
            simulationCV.wait(lk, [state, quit] { return *state == SIMULATE || *quit; });
            if (*quit)
                break;
            entityManager.step();
            simulationFrame++;
            *state = RENDER;
        }
        renderCV.notify_one();
    }
    *state = SIMULATION_END;
}

void Engine::gameLoop() {
    entityManager.start();
    bool quit = false;
    LoopState state = SIMULATE;
    std::thread simulationThread(&Engine::simulationLoop, this, &quit, &state);
    renderFrame = 0;

    while (!window.shouldClose()) {
        int width, height;
        window.getFramebufferSize(width, height);
        {
            std::unique_lock<std::mutex> lk(mutex);
            renderCV.wait(lk, [&state] { return state == RENDER; });
            renderer.render(&entityManager, width, height);
            // UI::UIData data{renderer->getScene(), renderer->getShaderManager()};
            // ui->render(data);
            state = SIMULATE;
            window.pollEvents();
        }
        simulationCV.notify_one();
        window.present();
        renderFrame++;
    }
    {
        std::unique_lock<std::mutex> lk(mutex);
        quit = true;
    }
    simulationCV.notify_one();
    simulationThread.join();
}
