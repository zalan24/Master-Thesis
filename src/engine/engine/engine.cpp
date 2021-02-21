#include "engine.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

#include <drv.h>
#include <drverror.h>
#include <drvwindow.h>

#include <irenderer.h>
#include <isimulation.h>

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

Engine::Engine(const std::string& configFile, const std::string& shaderbinFile,
               ResourceManager::ResourceInfos resource_infos)
  : Engine(get_config(configFile), shaderbinFile, std::move(resource_infos)) {
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

// Engine::SyncBlock::SyncBlock(drv::LogicalDevicePtr device) : imageAvailableSemaphore(device) {
// }

Engine::Engine(const Config& cfg, const std::string& shaderbinFile,
               ResourceManager::ResourceInfos resource_infos)
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
    // syncBlock(device),
    shaderBin(shaderbinFile),
    resourceMgr(std::move(resource_infos)) {
}

Engine::~Engine() {
    // entityManager.deleteAll();
}

void Engine::initGame(IRenderer* _renderer, ISimulation* _simulation) {
    simulation = _simulation;
    renderer = _renderer;
    inputSampleNode = frameGraph.addNode(FrameGraph::Node("sample_input"));
    // FrameGraph::NodeId simEntities =
    //   frameGraph.addNode(FrameGraph::Node("entities/simulate"));
    // FrameGraph::NodeId beforeDrawEntities =
    //   frameGraph.addNode(FrameGraph::Node("entities/beforeDraw"));
    // FrameGraph::NodeId recordEntities =
    //   frameGraph.addNode(FrameGraph::Node("entities/record"));
    // FrameGraph::NodeId clearGbuffer =
    //   frameGraph.addNode(FrameGraph::Node("gbuffer/clear"));
    // FrameGraph::NodeId resolveGbuffer =
    //   frameGraph.addNode(FrameGraph::Node("gbuffer/resolve"));
    presentFrameNode = frameGraph.addNode(FrameGraph::Node("presentFrame"));
    // These are just marker nodes
    // no actual work is done in them
    simStartNode = frameGraph.addNode(FrameGraph::Node("simulation/start"));
    simEndNode = frameGraph.addNode(FrameGraph::Node("simulation/end"));
    recordStartNode = frameGraph.addNode(FrameGraph::Node("record/start"));
    recordEndNode = frameGraph.addNode(FrameGraph::Node("record/end"));
    // FrameGraph::NodeId executeStart =
    //   frameGraph.addNode(FrameGraph::Node("execute/start"));
    // FrameGraph::NodeId executeEnd = frameGraph.addNode(FrameGraph::Node("execute/end"));

    FrameGraph::NodeDependency inputDep;
    inputDep.srcNode = inputSampleNode;

    // FrameGraph::NodeDependency entitySimDep;
    // entitySimDep.srcNode = simEntities;

    // TODO fill out offset fields
    // TODO review framegraph design (not complete)

    FrameGraph::NodeDependency simEnd_nextFrameDep;
    simEnd_nextFrameDep.srcNode = simEndNode;
    simEnd_nextFrameDep.cpu_cpuOffset = 1;

    FrameGraph::NodeDependency recEnd_nextFrameDep;
    recEnd_nextFrameDep.srcNode = recordEndNode;
    recEnd_nextFrameDep.cpu_cpuOffset = 1;

    FrameGraph::NodeDependency simStartDep;
    simStartDep.srcNode = simStartNode;

    FrameGraph::NodeDependency recordStartDep;
    recordStartDep.srcNode = recordStartNode;

    // FrameGraph::NodeDependency executionStartDep;
    // executionStartDep.srcNode = recordStart;

    FrameGraph::NodeDependency presentDep;
    presentDep.srcNode = presentFrameNode;

    // FrameGraph::NodeDependency gbufferClearDep;
    // gbufferClearDep.srcNode = clearGbuffer;
    // gbufferClearDep.cpu_cpuOffset = FrameGraph::NodeDependency::NO_SYNC;
    // gbufferClearDep.gpu_gpuOffset = 0;

    // FrameGraph::NodeDependency gbufferResolveDep;
    // gbufferResolveDep.srcNode = resolveGbuffer;
    // gbufferResolveDep.cpu_cpuOffset = FrameGraph::NodeDependency::NO_SYNC;
    // gbufferResolveDep.gpu_gpuOffset = 0;

    // TODO commented nodes should be moved to the renderer implementation
    // also, entity manager should not be part of the engine

    // FrameGraph::NodeDependency entityBeforeDrawDep;
    // entityBeforeDrawDep.srcNode = beforeDrawEntities;
    // entityBeforeDrawDep.gpu_gpuOffset = 0;

    // FrameGraph::NodeDependency entityRecord_entitySimDep;
    // entityRecord_entitySimDep.srcNode = recordEntities;
    // entityRecord_entitySimDep.cpu_cpuOffset = 1;

    // FrameGraph::NodeDependency entityRecord_gbufferResolveDep;
    // entityRecord_gbufferResolveDep.srcNode = recordEntities;
    // entityRecord_gbufferResolveDep.cpu_cpuOffset = FrameGraph::NodeDependency::NO_SYNC;
    // entityRecord_gbufferResolveDep.gpu_gpuOffset = 0;

    // FrameGraph::NodeDependency finalize_clearDep;
    // finalize_clearDep.srcNode = finalizeFrame;
    // finalize_clearDep.cpu_cpuOffset = FrameGraph::NodeDependency::NO_SYNC;
    // finalize_clearDep.gpu_gpuOffset = 1;

    // TODO wait for present at some point?

    // frameGraph.addDependency(simEntities, inputDep);
    // frameGraph.addDependency(beforeDrawEntities, entitySimDep);
    // frameGraph.addDependency(recordEntities, entityBeforeDrawDep);
    // frameGraph.addDependency(simEntities, entityRecord_entitySimDep);
    // frameGraph.addDependency(resolveGbuffer, entityRecord_gbufferResolveDep);
    // frameGraph.addDependency(recordEntities, gbufferClearDep);
    // frameGraph.addDependency(finalizeFrame, gbufferResolveDep);
    // frameGraph.addDependency(clearGbuffer, finalize_clearDep);
    // frameGraph.addDependency(simEnd, entitySimDep);
    frameGraph.addDependency(inputSampleNode, simEnd_nextFrameDep);
    frameGraph.addDependency(simStartNode, inputDep);
    frameGraph.addDependency(simStartNode, recEnd_nextFrameDep);
    frameGraph.addDependency(recordStartNode, simStartDep);
    frameGraph.addDependency(simEndNode, simStartDep);
    frameGraph.addDependency(presentFrameNode, recordStartDep);
    frameGraph.addDependency(recordEndNode, presentDep);

    // frameGraph.addDependency(inputSampleNode, simStartDep);
    // frameGraph.addDependency(simEnd, simStartDep);
    // frameGraph.addDependency(simEnd, inputDep);
    // frameGraph.addDependency(recordStart, simStartDep);  // can be parallel with sim
    // frameGraph.addDependency(presentFrame, recordStartDep);
    // frameGraph.addDependency(recordEnd, presentDep);
    // frameGraph.addDependency(executeStart, recordStartDep);  // can be parallel with record
    // frameGraph.addDependency(executeEnd, executionStartDep);

    ISimulation::FrameGraphData simData;
    simData.simStart = simStartNode;
    simData.sampleInput = inputSampleNode;
    simData.simEnd = simEndNode;

    IRenderer::FrameGraphData renderData;
    renderData.recStart = recordStartNode;
    renderData.recEnd = recordEndNode;
    renderData.present = presentFrameNode;

    simulation->initSimulationFrameGraph(frameGraph, simData);
    renderer->initRenderFrameGraph(frameGraph, renderData);

    frameGraph.validate();
}

bool Engine::sampleInput(FrameGraph::FrameId frameId) {
    FrameGraph::NodeHandle inputHandle = frameGraph.acquireNode(inputSampleNode, frameId);
    if (!inputHandle)
        return false;
    Input::InputEvent event;
    while (input.popEvent(event))
        inputManager.feedInput(std::move(event));
    return true;
}

// // TODO remove sync with record thread, rely on frameGraph instead
// void Engine::simulationLoop(RenderState* state) {
//     while (!state->quit) {
//         FrameId simulationFrame = state->simulationFrame.load();
//         {
//             std::unique_lock<std::mutex> lk(state->simulationMutex);
//             state->simulationCV.wait(lk, [state] { return state->canSimulate || state->quit; });
//             state->canSimulate = false;
//             if (state->quit)
//                 break;
//         }
//         // TODO latency sleep
//         sampleInput();
//         // entityManager.step();
//         std::this_thread::sleep_for(std::chrono::milliseconds(8));  // instead of simulation
//         {
//             std::unique_lock<std::mutex> lk(state->recordMutex);
//             state->canRecord = true;
//             state->recordCV.notify_one();
//         }
//         state->simulationFrame.fetch_add(FrameId(1));
//     }
// }

void Engine::simulationLoop() {
    FrameGraph::FrameId simulationFrame = 0;
    while (!frameGraph.isStopped()) {
        if (!sampleInput(simulationFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        {
            FrameGraph::NodeHandle simStartHandle =
              frameGraph.acquireNode(simStartNode, simulationFrame);
            if (!simStartHandle) {
                assert(frameGraph.isStopped());
                break;
            }
        }
        simulation->simulate(simulationFrame);
        {
            FrameGraph::NodeHandle simEndHandle =
              frameGraph.acquireNode(simEndNode, simulationFrame);
            if (!simEndHandle) {
                assert(frameGraph.isStopped());
                break;
            }
        }
        simulationFrame++;
    }
}

void Engine::present(FrameGraph::FrameId presentFrame) {
    // TODO
    // drv::PresentInfo info;
    // info.semaphoreCount = 1;
    // drv::SemaphorePtr semaphore = syncBlock.imageAvailableSemaphore;  // TODO
    // info.waitSemaphores = &semaphore;
    // drv::PresentResult result = swapchain.present(presentQueue.queue, info);
    // drv::drv_assert(result != drv::PresentResult::ERROR, "Present error");
    // if (result == drv::PresentResult::RECREATE_ADVISED
    //     || result == drv::PresentResult::RECREATE_REQUIRED) {
    //     state->recreateSwapchain = presentFrame;
    // }
}

// void Engine::recordCommandsLoop(RenderState* state) {
//     while (!state->quit) {
//         FrameId recordFrame = state->recordFrame.load();
//         // Do pre-record stuff (allocator, etc.)
//         uint32_t w = window->getWidth();
//         uint32_t h = window->getHeight();
//         if (state->swapchainCreated.load() < state->recreateSwapchain.load()
//             || w != swapchain.getCurrentWidth() || h != swapchain.getCurrentHeight()) {
//             state->swapchainCreated = recordFrame;
//             swapchain.recreate(physicalDevice, window);
//         }
//         {
//             std::unique_lock<std::mutex> lk(state->recordMutex);
//             state->recordCV.wait(lk, [state] { return state->canRecord || state->quit; });
//             state->canRecord = false;
//             if (state->quit)
//                 break;
//         }
//         // Stuff that depends on simulation, but not recorded yet
//         {
//             std::unique_lock<std::mutex> lk(state->recordMutex);
//             // TODO latency slop
//             state->recordCV.wait(lk, [state, recordFrame, this] {
//                 return recordFrame - state->executeFrame.load() + 1
//                          <= config.maxFramesInExecutionQueue
//                        || state->quit;
//             });
//             // ---
//             state->canRecord = false;
//             if (state->quit)
//                 break;
//         }
//         state->executionQueue->push(ExecutionPackage(ExecutionPackage::MessagePackage{
//           ExecutionPackage::Message::RECORD_START, state->recordFrame.load(), 0, nullptr}));
//         // TODO render entities
//         // entityManager.step();
//         std::this_thread::sleep_for(std::chrono::milliseconds(8));  // instead of rendering
//         {
//             std::unique_lock<std::mutex> lk(state->simulationMutex);
//             state->canSimulate = true;
//             state->simulationCV.notify_one();
//         }
//         // Do work here, that's unrelated to simulation

//         // TODO latency slop
//         bool acquiredSuccess = swapchain.acquire(syncBlock.imageAvailableSemaphore);
//         // ---
//         if (acquiredSuccess) {
//             state->executionQueue->push(ExecutionPackage(ExecutionPackage::MessagePackage{
//               ExecutionPackage::Message::PRESENT, state->recordFrame.load(), 0, nullptr}));
//         }

//         state->executionQueue->push(ExecutionPackage(
//           ExecutionPackage::MessagePackage{ExecutionPackage::Message::RECORD_END, 0, 0, nullptr}));

//         state->recordFrame.fetch_add(FrameId(1));
//     }
//     state->executionQueue->push(ExecutionPackage(
//       ExecutionPackage::MessagePackage{ExecutionPackage::Message::QUIT, 0, 0, nullptr}));
// }

void Engine::recordCommandsLoop() {
    FrameGraph::FrameId recordFrame = 0;
    while (!frameGraph.isStopped()) {
        // TODO preframe stuff (resize)
        // TODO wait on execution queue
        {
            FrameGraph::NodeHandle recStartHandle =
              frameGraph.acquireNode(recordStartNode, recordFrame);
            if (!recStartHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            // state->executionQueue->push(ExecutionPackage(ExecutionPackage::MessagePackage{
            //           ExecutionPackage::Message::RECORD_START, state->recordFrame.load(), 0, nullptr}));
        }
        renderer->record(recordFrame);
        {
            FrameGraph::NodeHandle presentHandle =
              frameGraph.acquireNode(presentFrameNode, presentFrameNode);
            if (!presentHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            //         // TODO latency slop
            //         bool acquiredSuccess = swapchain.acquire(syncBlock.imageAvailableSemaphore);
            //         // ---
            //         if (acquiredSuccess) {
            //             state->executionQueue->push(ExecutionPackage(ExecutionPackage::MessagePackage{
            //               ExecutionPackage::Message::PRESENT, state->recordFrame.load(), 0, nullptr}));
            //         }
        }
        {
            FrameGraph::NodeHandle recEndHandle =
              frameGraph.acquireNode(recordEndNode, recordFrame);
            if (!recEndHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            //         state->executionQueue->push(ExecutionPackage(
            //           ExecutionPackage::MessagePackage{ExecutionPackage::Message::RECORD_END, 0, 0, nullptr}));
        }
        recordFrame++;
    }
    // state->executionQueue->push(ExecutionPackage(
    //   ExecutionPackage::MessagePackage{ExecutionPackage::Message::QUIT, 0, 0, nullptr}));
}

void Engine::executeCommandsLoop() {
    // TODO
    // while (true) {
    //     ExecutionPackage package;
    //     state->executionQueue->waitForPackage();
    //     while (state->executionQueue->pop(package)) {
    //         if (std::holds_alternative<ExecutionPackage::MessagePackage>(package.package)) {
    //             ExecutionPackage::MessagePackage& message =
    //               std::get<ExecutionPackage::MessagePackage>(package.package);
    //             switch (message.msg) {
    //                 case ExecutionPackage::Message::RECORD_START: {
    //                     std::unique_lock<std::mutex> lk(state->recordMutex);
    //                     state->executeFrame.store(message.value1);
    //                     state->recordCV.notify_one();
    //                 } break;
    //                 case ExecutionPackage::Message::RECORD_END:
    //                     // TODO latency slop: measure time between execution RECORD_END (here) and time of recording first command for next frame
    //                     // It's possible that record end is executed before first next command (nothing to do then)
    //                     // first command doesn't need to be record start (first functor or cmd buffer???)
    //                     break;
    //                 case ExecutionPackage::Message::PRESENT:
    //                     present(state, message.value1);
    //                     break;
    //                 case ExecutionPackage::Message::QUIT:
    //                     return;
    //             }
    //         }
    //         else if (std::holds_alternative<ExecutionPackage::Functor>(package.package)) {
    //             ExecutionPackage::Functor& functor =
    //               std::get<ExecutionPackage::Functor>(package.package);
    //             functor();
    //         }
    //         else if (std::holds_alternative<ExecutionPackage::CommandBufferPackage>(
    //                    package.package)) {
    //             ExecutionPackage::CommandBufferPackage& cmdBuffer =
    //               std::get<ExecutionPackage::CommandBufferPackage>(package.package);
    //             drv::drv_assert(false, "Not implemented");
    //         }
    //     }
    // }
}

void Engine::gameLoop() {
    if (simulation == nullptr || renderer == nullptr)
        throw std::runtime_error("Simulation or renderer was not initialized before game loop");
    // entityManager.start();
    // ExecutionQueue executionQueue;
    // RenderState state;
    // state.executionQueue = &executionQueue;
    std::thread simulationThread(&Engine::simulationLoop, this);
    std::thread recordThread(&Engine::recordCommandsLoop, this);
    // std::thread executeThread(&Engine::executeCommandsLoop, this, &state);

    set_thread_name(&simulationThread, "simulation");
    set_thread_name(&recordThread, "record");
    // set_thread_name(&executeThread, "execute");

    IWindow* w = window;
    while (!w->shouldClose()) {
        // TODO this sleep could be replaced with a sync with simulation
        // only need this data for input sampling
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        static_cast<IWindow*>(window)->pollEvents();
    }
    // {
    // std::unique_lock<std::mutex> simLk(state.simulationMutex);
    // std::unique_lock<std::mutex> recLk(state.recordMutex);
    // state.quit = true;
    // state.simulationCV.notify_one();
    // state.recordCV.notify_one();
    frameGraph.stopExecution();
    // }
    simulationThread.join();
    recordThread.join();
    // executeThread.join();
}
