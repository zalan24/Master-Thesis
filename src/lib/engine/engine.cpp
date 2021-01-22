#include "engine.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

#include <drv.h>
#include <drverror.h>
#include <drvwindow.h>

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

Engine::Engine(const std::string& configFile) : Engine(get_config(configFile)) {
}

Engine::Engine(const Config& cfg)
  : config(cfg),
    driver({get_driver(cfg.driver)}),
    drvInstance(drv::InstanceCreateInfo{cfg.title.c_str()}),
    physicalDevice(get_device_selection_info(drvInstance)),
    commandLaneMgr(
      physicalDevice,
      {{"main",
        {{"render", 0.5, drv::CMD_TYPE_GRAPHICS, drv::CMD_TYPE_COMPUTE | drv::CMD_TYPE_TRANSFER},
         {"compute", 0.5, drv::CMD_TYPE_COMPUTE, drv::CMD_TYPE_TRANSFER},
         {"DtoH", 0.5, drv::CMD_TYPE_TRANSFER, 0},
         {"HtoD", 0.5, drv::CMD_TYPE_TRANSFER, 0}}},
       {"input", {{"HtoD", 1, drv::CMD_TYPE_TRANSFER, 0}}}}),
    device({physicalDevice, commandLaneMgr.getQueuePriorityInfo()}),
    queueManager(device, &commandLaneMgr),
    renderQueue(queueManager.getQueue({"main", "render"})),
    computeQueue(queueManager.getQueue({"main", "compute"})),
    DtoHQueue(queueManager.getQueue({"main", "DtoH"})),
    HtoDQueue(queueManager.getQueue({"main", "HtoD"})),
    inputQueue(queueManager.getQueue({"input", "HtoD"})),
    cmdBufferBank(device),
    window(drv::WindowOptions{static_cast<unsigned int>(cfg.screenWidth),
                              static_cast<unsigned int>(cfg.screenHeight), cfg.title.c_str()}) {
}

Engine::~Engine() {
}

void Engine::simulationLoop(RenderState* state) {
    while (!state->quit) {
        FrameId simulationFrame = state->simulationFrame.load();
        {
            std::unique_lock<std::mutex> lk(state->simulationMutex);
            state->simulationCV.wait(lk, [state] { return state->canSimulate || state->quit; });
            state->canSimulate = false;
            if (state->quit)
                break;
        }
        // TODO wait for render (max frames)
        entityManager.step();
        {
            std::unique_lock<std::mutex> lk(state->recordMutex);
            state->canRecord = true;
            state->recordCV.notify_one();
        }
        state->simulationFrame.fetch_add(FrameId(1));
    }
    // *state = SIMULATION_END;
}

void Engine::recordCommandsLoop(RenderState* state) {
    while (!state->quit) {
        FrameId recordFrame = state->recordFrame.load();
        // Do pre-record stuff (allocator, etc.)
        {
            std::unique_lock<std::mutex> lk(state->recordMutex);
            state->recordCV.wait(lk, [state] { return state->canRecord || state->quit; });
            state->canRecord = false;
            if (state->quit)
                break;
        }
        // TODO render entities
        // entityManager.step();
        {
            std::unique_lock<std::mutex> lk(state->simulationMutex);
            state->canSimulate = true;
            state->simulationCV.notify_one();
        }
        // Do work here, that's unrelated to simulation
        state->recordFrame.fetch_add(FrameId(1));
    }
}

void Engine::executeCommandsLoop(RenderState* state) {
    // while (!state->quit) {
    //     FrameId executeFrame = state->executeFrame.load();
    //     {
    //         std::unique_lock<std::mutex> lk(state->executeMutex);
    //         state->executeCV.wait(lk, [state] { return state->canExecute || state->quit; });
    //         state->canExecute = false;
    //         if (state->quit)
    //             break;
    //     }

    //     state->executeFrame.fetch_add(FrameId(1));
    // }
}

void Engine::gameLoop() {
    entityManager.start();
    RenderState state;
    std::thread simulationThread(&Engine::simulationLoop, this, &state);
    std::thread recordThread(&Engine::recordCommandsLoop, this, &state);
    std::thread executeThread(&Engine::executeCommandsLoop, this, &state);

    IWindow* w = window;
    // while (!w->shouldClose()) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // unsigned int width, height;
    // w->getContentSize(width, height);
    // {
    //     std::unique_lock<std::mutex> lk(mutex);
    //     renderCV.wait(lk, [&state] { return state == RENDER; });
    //     renderer.render(&entityManager, width, height);
    //     // UI::UIData data{renderer->getScene(), renderer->getShaderManager()};
    //     // ui->render(data);
    //     state = SIMULATE;
    //     window.pollEvents();
    // }
    // simulationCV.notify_one();
    // window.present();
    // renderFrame++;
    // }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::unique_lock<std::mutex> simLk(state.simulationMutex);
        std::unique_lock<std::mutex> recLk(state.recordMutex);
        state.quit = true;
        state.simulationCV.notify_one();
        state.recordCV.notify_one();
    }
    simulationThread.join();
    recordThread.join();
    executeThread.join();
}
