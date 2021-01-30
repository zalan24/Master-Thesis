#include "engine.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

#include <drv.h>
#include <drverror.h>
#include <drvwindow.h>

#include <namethreads.h>
// #include <rendercontext.h>

#include "execution_queue.h"

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

void Engine::Config::writeJson(json& out) const {
    WRITE_OBJECT(screenWidth, out);
    WRITE_OBJECT(screenHeight, out);
    WRITE_OBJECT(stackMemorySizeKb, out);
    WRITE_OBJECT(inputBufferSize, out);
    WRITE_OBJECT(title, out);
    WRITE_OBJECT(imagesInSwapchain, out);
    WRITE_OBJECT(maxFramesInExecutionQueue, out);
    WRITE_OBJECT(maxFramesOnGPU, out);
    WRITE_OBJECT(driver, out);
}

void Engine::Config::readJson(const json& in) {
    READ_OBJECT(screenWidth, in);
    READ_OBJECT(screenHeight, in);
    READ_OBJECT(stackMemorySizeKb, in);
    READ_OBJECT(inputBufferSize, in);
    READ_OBJECT(title, in);
    READ_OBJECT(imagesInSwapchain, in);
    READ_OBJECT(maxFramesInExecutionQueue, in);
    READ_OBJECT(maxFramesOnGPU, in);
    READ_OBJECT(driver, in);
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

drv::PhysicalDevice::SelectionInfo Engine::get_device_selection_info(
  drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions) {
    drv::PhysicalDevice::SelectionInfo selectInfo;
    selectInfo.instance = instance;
    selectInfo.requirePresent = true;
    selectInfo.extensions = deviceExtensions;
    selectInfo.compare = drv::PhysicalDevice::pick_discere_card;
    selectInfo.commandMasks = {drv::CMD_TYPE_TRANSFER, drv::CMD_TYPE_COMPUTE,
                               drv::CMD_TYPE_GRAPHICS};
    return selectInfo;
}

Engine::ErrorCallback::ErrorCallback() {
    drv::set_callback(::callback);
}

Engine::WindowIniter::WindowIniter(IWindow* _window, drv::InstancePtr instance) : window(_window) {
    drv::drv_assert(window->init(instance), "Could not initialize window");
}

Engine::WindowIniter::~WindowIniter() {
    window->close();
}

Engine::Engine(const std::string& configFile, ResourceManager::ResourceInfos resource_infos)
  : Engine(get_config(configFile), std::move(resource_infos)) {
}

drv::Swapchain::CreateInfo Engine::get_swapchain_create_info(const Config& config) {
    drv::Swapchain::CreateInfo ret;
    ret.clipped = true;
    ret.preferredImageCount = static_cast<uint32_t>(config.imagesInSwapchain);
    ret.formatPreferences = {drv::ImageFormat::B8G8R8A8_SRGB};
    ret.preferredPresentModes = {drv::SwapchainCreateInfo::PresentMode::MAILBOX,
                                 drv::SwapchainCreateInfo::PresentMode::IMMEDIATE,
                                 drv::SwapchainCreateInfo::PresentMode::FIFO_RELAXED,
                                 drv::SwapchainCreateInfo::PresentMode::FIFO};
    return ret;
}

Engine::SyncBlock::SyncBlock(drv::LogicalDevicePtr device) : imageAvailableSemaphore(device) {
}

Engine::Engine(const Config& cfg, ResourceManager::ResourceInfos resource_infos)
  : config(cfg),
    coreContext({size_t(config.stackMemorySizeKb << 10)}),
    input(static_cast<size_t>(config.inputBufferSize)),
    driver({get_driver(cfg.driver)}),
    window(&input, &inputManager,
           drv::WindowOptions{static_cast<unsigned int>(cfg.screenWidth),
                              static_cast<unsigned int>(cfg.screenHeight), cfg.title.c_str()}),
    drvInstance(drv::InstanceCreateInfo{cfg.title.c_str()}),
    windowIniter(window, drvInstance),
    deviceExtensions(true),
    physicalDevice(get_device_selection_info(drvInstance, deviceExtensions), window),
    commandLaneMgr(
      physicalDevice, window,
      {{"main",
        {{"render", 0.5, drv::CMD_TYPE_GRAPHICS, drv::CMD_TYPE_COMPUTE | drv::CMD_TYPE_TRANSFER,
          false, true},
         {"present", 0.5, 0,
          drv::CMD_TYPE_COMPUTE | drv::CMD_TYPE_TRANSFER | drv::CMD_TYPE_GRAPHICS, true, false},
         {"compute", 0.5, drv::CMD_TYPE_COMPUTE, drv::CMD_TYPE_TRANSFER, false, true},
         {"DtoH", 0.5, drv::CMD_TYPE_TRANSFER, 0, false, true},
         {"HtoD", 0.5, drv::CMD_TYPE_TRANSFER, 0, false, true}}},
       {"input", {{"HtoD", 1, drv::CMD_TYPE_TRANSFER, 0, false, true}}}}),
    device({physicalDevice, commandLaneMgr.getQueuePriorityInfo(), deviceExtensions}),
    queueManager(device, &commandLaneMgr),
    renderQueue(queueManager.getQueue({"main", "render"})),
    presentQueue(queueManager.getQueue({"main", "present"})),
    computeQueue(queueManager.getQueue({"main", "compute"})),
    DtoHQueue(queueManager.getQueue({"main", "DtoH"})),
    HtoDQueue(queueManager.getQueue({"main", "HtoD"})),
    inputQueue(queueManager.getQueue({"input", "HtoD"})),
    cmdBufferBank(device),
    swapchain(physicalDevice, device, window, get_swapchain_create_info(config)),
    syncBlock(device),
    resourceMgr(std::move(resource_infos)) {
}

Engine::~Engine() {
    entityManager.deleteAll();
}

void Engine::sampleInput() {
    drv::Input::InputEvent event;
    while (input.popEvent(event))
        inputManager.feedInput(std::move(event));
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
        // TODO latency sleep
        sampleInput();
        entityManager.step();
        std::this_thread::sleep_for(std::chrono::milliseconds(8));  // instead of simulation
        {
            std::unique_lock<std::mutex> lk(state->recordMutex);
            state->canRecord = true;
            state->recordCV.notify_one();
        }
        state->simulationFrame.fetch_add(FrameId(1));
    }
}

void Engine::present(RenderState* state, FrameId presentFrame) {
    drv::PresentInfo info;
    info.semaphoreCount = 1;
    drv::SemaphorePtr semaphore = syncBlock.imageAvailableSemaphore;  // TODO
    info.waitSemaphores = &semaphore;
    drv::PresentResult result = swapchain.present(presentQueue.queue, info);
    drv::drv_assert(result != drv::PresentResult::ERROR, "Present error");
    if (result == drv::PresentResult::RECREATE_ADVISED
        || result == drv::PresentResult::RECREATE_REQUIRED) {
        state->recreateSwapchain = presentFrame;
    }
}

void Engine::recordCommandsLoop(RenderState* state) {
    while (!state->quit) {
        FrameId recordFrame = state->recordFrame.load();
        // Do pre-record stuff (allocator, etc.)
        uint32_t w = window->getWidth();
        uint32_t h = window->getHeight();
        if (state->swapchainCreated.load() < state->recreateSwapchain.load()
            || w != swapchain.getCurrentWidth() || h != swapchain.getCurrentHeight()) {
            state->swapchainCreated = recordFrame;
            swapchain.recreate(physicalDevice, window);
        }
        {
            std::unique_lock<std::mutex> lk(state->recordMutex);
            state->recordCV.wait(lk, [state] { return state->canRecord || state->quit; });
            state->canRecord = false;
            if (state->quit)
                break;
        }
        // Stuff that depends on simulation, but not recorded yet
        {
            std::unique_lock<std::mutex> lk(state->recordMutex);
            // TODO latency slop
            state->recordCV.wait(lk, [state, recordFrame, this] {
                return recordFrame - state->executeFrame.load() + 1
                         <= config.maxFramesInExecutionQueue
                       || state->quit;
            });
            // ---
            state->canRecord = false;
            if (state->quit)
                break;
        }
        state->executionQueue->push(ExecutionPackage(ExecutionPackage::MessagePackage{
          ExecutionPackage::Message::RECORD_START, state->recordFrame.load(), 0, nullptr}));
        // TODO render entities
        // entityManager.step();
        std::this_thread::sleep_for(std::chrono::milliseconds(8));  // instead of rendering
        {
            std::unique_lock<std::mutex> lk(state->simulationMutex);
            state->canSimulate = true;
            state->simulationCV.notify_one();
        }
        // Do work here, that's unrelated to simulation

        // TODO latency slop
        bool acquiredSuccess = swapchain.acquire(syncBlock.imageAvailableSemaphore);
        // ---
        if (acquiredSuccess) {
            state->executionQueue->push(ExecutionPackage(ExecutionPackage::MessagePackage{
              ExecutionPackage::Message::PRESENT, state->recordFrame.load(), 0, nullptr}));
        }

        state->executionQueue->push(ExecutionPackage(
          ExecutionPackage::MessagePackage{ExecutionPackage::Message::RECORD_END, 0, 0, nullptr}));

        state->recordFrame.fetch_add(FrameId(1));
    }
    state->executionQueue->push(ExecutionPackage(
      ExecutionPackage::MessagePackage{ExecutionPackage::Message::QUIT, 0, 0, nullptr}));
}

void Engine::executeCommandsLoop(RenderState* state) {
    while (true) {
        ExecutionPackage package;
        state->executionQueue->waitForPackage();
        while (state->executionQueue->pop(package)) {
            if (std::holds_alternative<ExecutionPackage::MessagePackage>(package.package)) {
                ExecutionPackage::MessagePackage& message =
                  std::get<ExecutionPackage::MessagePackage>(package.package);
                switch (message.msg) {
                    case ExecutionPackage::Message::RECORD_START: {
                        std::unique_lock<std::mutex> lk(state->recordMutex);
                        state->executeFrame.store(message.value1);
                        state->recordCV.notify_one();
                    } break;
                    case ExecutionPackage::Message::RECORD_END:
                        // TODO latency slop: measure time between execution RECORD_END (here) and time of recording first command for next frame
                        // It's possible that record end is executed before first next command (nothing to do then)
                        // first command doesn't need to be record start (first functor or cmd buffer???)
                        break;
                    case ExecutionPackage::Message::PRESENT:
                        present(state, message.value1);
                        break;
                    case ExecutionPackage::Message::QUIT:
                        return;
                }
            }
            else if (std::holds_alternative<ExecutionPackage::Functor>(package.package)) {
                ExecutionPackage::Functor& functor =
                  std::get<ExecutionPackage::Functor>(package.package);
                functor();
            }
            else if (std::holds_alternative<ExecutionPackage::CommandBufferPackage>(
                       package.package)) {
                ExecutionPackage::CommandBufferPackage& cmdBuffer =
                  std::get<ExecutionPackage::CommandBufferPackage>(package.package);
                drv::drv_assert(false, "Not implemented");
            }
        }
    }
}

void Engine::gameLoop() {
    entityManager.start();
    ExecutionQueue executionQueue;
    RenderState state;
    state.executionQueue = &executionQueue;
    std::thread simulationThread(&Engine::simulationLoop, this, &state);
    std::thread recordThread(&Engine::recordCommandsLoop, this, &state);
    std::thread executeThread(&Engine::executeCommandsLoop, this, &state);

    set_thread_name(&simulationThread, "simulation");
    set_thread_name(&recordThread, "record");
    set_thread_name(&executeThread, "execute");

    IWindow* w = window;
    while (!w->shouldClose()) {
        // TODO this sleep could be replaced with a sync with simulation
        // only need this data for input sampling
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        static_cast<IWindow*>(window)->pollEvents();
    }
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
