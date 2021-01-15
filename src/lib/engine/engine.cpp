#include "engine.h"

#include <fstream>

// #include <rendercontext.h>

void Engine::Config::gatherEntries(std::vector<ISerializable::Entry>& entries) const {
    REGISTER_ENTRY(screenWidth, entries);
    REGISTER_ENTRY(screenHeight, entries);
    REGISTER_ENTRY(title, entries);
}

static Engine::Config get_config(const std::string& file) {
    std::ifstream in(file);
    assert(in.is_open());
    Engine::Config config;
    config.read(in);
    return config;
}

Engine::Engine(const std::string& configFile) : Engine(get_config(configFile)) {
}

Engine::Engine(const Config& cfg)
  : config(cfg)  //, window(config.screenWidth, config.screenHeight, config.title)
{
}

Engine::~Engine() {
    // TODO
}

void Engine::simulationLoop(volatile bool* quit, volatile LoopState* state) {
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
    volatile bool quit = false;
    volatile LoopState state = SIMULATE;
    std::thread simulationThread(&Engine::simulationLoop, this, &quit, &state);
    renderFrame = 0;

    // while (!window.shouldClose()) {
    //     int width, height;
    //     window.getFramebufferSize(width, height);
    //     {
    //         std::unique_lock<std::mutex> lk(mutex);
    //         renderCV.wait(lk, [&state] { return state == RENDER; });
    //         renderer.render(&entityManager, width, height);
    //         // UI::UIData data{renderer->getScene(), renderer->getShaderManager()};
    //         // ui->render(data);
    //         state = SIMULATE;
    //         window.pollEvents();
    //     }
    //     simulationCV.notify_one();
    //     window.present();
    //     renderFrame++;
    // }
    {
        std::unique_lock<std::mutex> lk(mutex);
        quit = true;
    }
    simulationCV.notify_one();
    simulationThread.join();
}
