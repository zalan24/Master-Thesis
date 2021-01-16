#include "engine.h"

#include <fstream>
#include <iostream>
#include <map>

#include <drv.h>
#include <drverror.h>

// #include <rendercontext.h>

static void callback(const drv::CallbackData* data) {
    switch (data->type) {
        case drv::CallbackData::Type::VERBOSE:
            break;
        case drv::CallbackData::Type::NOTE:
            std::cout << data->text << std::endl;
            break;
        case drv::CallbackData::Type::WARNING:
            std::cerr << data->text << std::endl;
            break;
        case drv::CallbackData::Type::ERROR:
            std::cerr << data->text << std::endl;
            throw std::runtime_error(std::string{data->text});
        case drv::CallbackData::Type::FATAL:
            std::cerr << data->text << std::endl;
            std::abort();
    }
}

void Engine::Config::gatherEntries(std::vector<ISerializable::Entry>& entries) const {
    REGISTER_ENTRY(screenWidth, entries);
    REGISTER_ENTRY(screenHeight, entries);
    REGISTER_ENTRY(title, entries);
    REGISTER_ENTRY(driver, entries);
}

static Engine::Config get_config(const std::string& file) {
    std::ifstream in(file);
    assert(in.is_open());
    Engine::Config config;
    config.read(in);
    return config;
}

static drv::Driver get_driver(const std::string& name) {
    if (name == "Vulkan")
        return drv::Driver::VULKAN;
    throw std::runtime_error("Unknown driver: " + name);
}

drv::PhysicalDevice::SelectionInfo Engine::get_device_selection_info(drv::InstancePtr instance) {
    drv::PhysicalDevice::SelectionInfo selectInfo;
    selectInfo.instance = instance;
    selectInfo.compare = drv::PhysicalDevice::pick_discere_card;
    selectInfo.commandMasks = {drv::CMD_TYPE_TRANSFER, drv::CMD_TYPE_COMPUTE,
                               drv::CMD_TYPE_GRAPHICS};
    return selectInfo;
}

Engine::DriverSelector::DriverSelector(drv::Driver d) {
    drv::set_callback(callback);
    if (!drv::register_driver(&d, 1))
        throw std::runtime_error("Could not initialize driver");
}

Engine::Engine(const std::string& configFile) : Engine(get_config(configFile)) {
}

Engine::Engine(const Config& cfg)
  : config(cfg),
    driverSelector(get_driver(cfg.driver)),
    drvInstance(drv::InstanceCreateInfo{cfg.title.c_str()}),
    physicalDevice(get_device_selection_info(drvInstance)) {
}

Engine::~Engine() {
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
