#include "engine.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

#include <corecontext.h>

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
    WRITE_OBJECT(maxFramesInFlight, out);
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
    READ_OBJECT(maxFramesInFlight, in);
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

Engine::SyncBlock::SyncBlock(drv::LogicalDevicePtr device, uint32_t maxFramesInFlight) {
    imageAvailableSemaphores.reserve(maxFramesInFlight);
    renderFinishedSemaphores.reserve(maxFramesInFlight);
    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        imageAvailableSemaphores.emplace_back(device);
        renderFinishedSemaphores.emplace_back(device);
    }
}

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
    eventPool(device),
    syncBlock(device, config.maxFramesInFlight),
    shaderBin(shaderbinFile),
    resourceMgr(std::move(resource_infos)),
    frameGraph(device) {
}

Engine::~Engine() {
    // entityManager.deleteAll();
}

void Engine::initGame(IRenderer* _renderer, ISimulation* _simulation) {
    simulation = _simulation;
    renderer = _renderer;

    queueInfos.computeQueue = {computeQueue.queue, frameGraph.registerQueue(computeQueue.queue)};
    queueInfos.DtoHQueue = {DtoHQueue.queue, frameGraph.registerQueue(DtoHQueue.queue)};
    queueInfos.HtoDQueue = {HtoDQueue.queue, frameGraph.registerQueue(HtoDQueue.queue)};
    queueInfos.inputQueue = {inputQueue.queue, frameGraph.registerQueue(inputQueue.queue)};
    queueInfos.presentQueue = {presentQueue.queue, frameGraph.registerQueue(presentQueue.queue)};
    queueInfos.renderQueue = {renderQueue.queue, frameGraph.registerQueue(renderQueue.queue)};

    inputSampleNode = frameGraph.addNode(FrameGraph::Node("sample_input", false));
    presentFrameNode = frameGraph.addNode(FrameGraph::Node("presentFrame", true));
    cleanUpNode = frameGraph.addNode(FrameGraph::Node("cleanUp", false));
    // These are just marker nodes, no actual work is done in them
    simStartNode = frameGraph.addNode(FrameGraph::Node("simulation/start", false));
    simEndNode = frameGraph.addNode(FrameGraph::Node("simulation/end", false));
    recordStartNode = frameGraph.addNode(FrameGraph::Node("record/start", true));
    recordEndNode = frameGraph.addNode(FrameGraph::Node("record/end", true));
    executeStartNode = frameGraph.addNode(FrameGraph::Node("execute/start", false));
    executeEndNode = frameGraph.addNode(FrameGraph::Node("execute/end", false));

    if (config.maxFramesInExecutionQueue < 1)
        throw std::runtime_error("maxFramesInExecutionQueue must be at least 1");
    const uint32_t executionDepOffset = static_cast<uint32_t>(config.maxFramesInExecutionQueue);
    if (config.maxFramesInFlight < 2)
        throw std::runtime_error("maxFramesInFlight must be at least 2");
    if (config.maxFramesInFlight < config.maxFramesInExecutionQueue)
        throw std::runtime_error(
          "maxFramesInFlight must be at least the value of maxFramesInExecutionQueue");
    const uint32_t presentDepOffset = static_cast<uint32_t>(config.maxFramesInFlight - 1);
    const uint32_t cleanUpCpuOffset = static_cast<uint32_t>(config.maxFramesInFlight + 1);

    frameGraph.addDependency(inputSampleNode, FrameGraph::CpuDependency{simStartNode, 0});
    frameGraph.addDependency(simStartNode, FrameGraph::CpuDependency{simEndNode, 1});
    frameGraph.addDependency(simStartNode, FrameGraph::CpuDependency{recordEndNode, 1});
    frameGraph.addDependency(recordStartNode, FrameGraph::CpuDependency{simStartNode, 0});
    frameGraph.addDependency(recordStartNode,
                             FrameGraph::CpuDependency{executeEndNode, executionDepOffset});
    frameGraph.addDependency(recordStartNode, FrameGraph::EnqueueDependency{recordEndNode, 1});
    frameGraph.addDependency(simEndNode, FrameGraph::CpuDependency{simStartNode, 0});
    frameGraph.addDependency(simEndNode, FrameGraph::CpuDependency{inputSampleNode, 0});
    frameGraph.addDependency(presentFrameNode, FrameGraph::CpuDependency{recordStartNode, 0});
    frameGraph.addDependency(presentFrameNode, FrameGraph::EnqueueDependency{recordStartNode, 0});
    frameGraph.addDependency(recordEndNode, FrameGraph::CpuDependency{presentFrameNode, 0});
    frameGraph.addDependency(recordEndNode, FrameGraph::EnqueueDependency{presentFrameNode, 0});
    frameGraph.addDependency(executeStartNode, FrameGraph::CpuDependency{recordStartNode, 0});
    frameGraph.addDependency(executeEndNode, FrameGraph::CpuDependency{executeStartNode, 0});
    frameGraph.addDependency(
      presentFrameNode, FrameGraph::QueueCpuDependency{presentFrameNode, queueInfos.presentQueue.id,
                                                       presentDepOffset});
    frameGraph.addDependency(
      cleanUpNode, FrameGraph::QueueCpuDependency{presentFrameNode, queueInfos.presentQueue.id, 0});
    frameGraph.addDependency(cleanUpNode, FrameGraph::CpuDependency{executeEndNode, 0});
    frameGraph.addDependency(recordStartNode,
                             FrameGraph::CpuDependency{cleanUpNode, cleanUpCpuOffset});

    ISimulation::FrameGraphData simData;
    simData.simStart = simStartNode;
    simData.sampleInput = inputSampleNode;
    simData.simEnd = simEndNode;

    IRenderer::FrameGraphData renderData;
    renderData.recStart = recordStartNode;
    renderData.recEnd = recordEndNode;
    renderData.present = presentFrameNode;

    simulation->initSimulationFrameGraph(frameGraph, simData);
    FrameGraph::NodeId renderNode = FrameGraph::INVALID_NODE;
    FrameGraph::QueueId renderQueueId;
    if (renderer->initRenderFrameGraph(frameGraph, renderData, renderNode, renderQueueId)) {
        assert(renderNode != FrameGraph::INVALID_NODE);
        frameGraph.addDependency(presentFrameNode,
                                 FrameGraph::QueueQueueDependency{renderNode, renderQueueId,
                                                                  queueInfos.presentQueue.id, 0});
        frameGraph.addDependency(presentFrameNode, FrameGraph::EnqueueDependency{renderNode, 0});
    }

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

void Engine::simulationLoop(volatile std::atomic<FrameGraph::FrameId>* simulationFrame,
                            const volatile std::atomic<FrameGraph::FrameId>* stopFrame) {
    simulationFrame->store(0);
    while (!frameGraph.isStopped() && *simulationFrame <= *stopFrame) {
        {
            FrameGraph::NodeHandle simStartHandle =
              frameGraph.acquireNode(simStartNode, *simulationFrame);
            if (!simStartHandle) {
                assert(frameGraph.isStopped());
                break;
            }
        }
        if (!sampleInput(*simulationFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        simulation->simulate(frameGraph, *simulationFrame);
        {
            FrameGraph::NodeHandle simEndHandle =
              frameGraph.acquireNode(simEndNode, *simulationFrame);
            if (!simEndHandle) {
                assert(frameGraph.isStopped());
                break;
            }
        }
        {
            std::shared_lock<std::shared_mutex> lock(stopFrameMutex);
            simulationFrame->store(simulationFrame->load() + 1);
        }
    }
}

void Engine::present(FrameGraph::FrameId presentFrame) {
    drv::PresentInfo info;
    info.semaphoreCount = 1;
    drv::SemaphorePtr semaphore = syncBlock.renderFinishedSemaphores[acquireImageSemaphoreId];
    info.waitSemaphores = &semaphore;
    drv::PresentResult result = swapchain.present(presentQueue.queue, info);
    TODO;  // what's gonna wait on this?
    drv::drv_assert(result != drv::PresentResult::ERROR, "Present error");
    if (result == drv::PresentResult::RECREATE_ADVISED
        || result == drv::PresentResult::RECREATE_REQUIRED) {
        throw std::runtime_error("Implement swapchain recreation");
        // state->recreateSwapchain = presentFrame;
    }
}

const Engine::QueueInfo& Engine::getQueues() const {
    return queueInfos;
}

Engine::AcquiredImageData Engine::acquiredSwapchainImage(
  FrameGraph::NodeHandle& acquiringNodeHandle) {
    // TODO latency slop -> acquiringNodeHandle
    bool acquiredSuccess =
      swapchain.acquire(syncBlock.imageAvailableSemaphores[acquireImageSemaphoreId]);
    // ---
    Engine::AcquiredImageData ret;
    if (acquiredSuccess) {
        ret.image = swapchain.getAcquiredImage();
        ret.imageAvailableSemaphore = syncBlock.imageAvailableSemaphores[acquireImageSemaphoreId];
        ret.renderFinishedSemaphore = syncBlock.renderFinishedSemaphores[acquireImageSemaphoreId];
    }
    else {
        ret.image = drv::NULL_HANDLE;
        ret.imageAvailableSemaphore = drv::NULL_HANDLE;
        ret.renderFinishedSemaphore = drv::NULL_HANDLE;
    }
    acquireImageSemaphoreId =
      (acquireImageSemaphoreId + 1) % syncBlock.imageAvailableSemaphores.size();
    return ret;
}

Engine::CommandBufferRecorder Engine::acquireCommandRecorder(
  FrameGraph::NodeHandle& acquiringNodeHandle, FrameGraph::FrameId frameId,
  FrameGraph::QueueId queueId) {
    drv::QueuePtr queue = frameGraph.getQueue(queueId);
    drv::CommandBufferBankGroupInfo acquireInfo(drv::get_queue_family(device, queue), false,
                                                drv::CommandBufferType::PRIMARY);
    std::unique_lock<std::mutex> lock = drv::lock_queue(device, queue);
    return CommandBufferRecorder(std::move(lock), queue, queueId, &frameGraph, this,
                                 &acquiringNodeHandle, frameId, cmdBufferBank.acquire(acquireInfo));
}

void Engine::recordCommandsLoop(const volatile std::atomic<FrameGraph::FrameId>* stopFrame) {
    FrameGraph::FrameId recordFrame = 0;
    while (!frameGraph.isStopped() && recordFrame <= *stopFrame) {
        // TODO preframe stuff (resize)
        std::unique_ptr<Garbage> garbage;
        {
            FrameGraph::NodeHandle recStartHandle =
              frameGraph.acquireNode(recordStartNode, recordFrame);
            if (!recStartHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            garbage = std::make_unique<Garbage>(recordFrame);  // TODO create in frame mem
            frameGraph.getExecutionQueue(recStartHandle)
              ->push(ExecutionPackage(ExecutionPackage::MessagePackage{
                ExecutionPackage::Message::RECORD_START, recordFrame, 0,
                static_cast<void*>(garbage.get())}));
        }
        renderer->record(frameGraph, recordFrame);
        {
            FrameGraph::NodeHandle presentHandle =
              frameGraph.acquireNode(presentFrameNode, recordFrame);
            if (!presentHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            if (swapchain.getAcquiredImage() != drv::NULL_HANDLE) {
                frameGraph.getExecutionQueue(presentHandle)
                  ->push(ExecutionPackage(ExecutionPackage::MessagePackage{
                    ExecutionPackage::Message::PRESENT, recordFrame, 0, nullptr}));
            }
        }
        {
            FrameGraph::NodeHandle recEndHandle =
              frameGraph.acquireNode(recordEndNode, recordFrame);
            if (!recEndHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            struct ResetGarbageFunctor : ExecutionPackage::CustomFunctor
            {
                moodycamel::ConcurrentQueue<std::unique_ptr<Garbage>>* garbageQueue;
                std::unique_ptr<Garbage> garb;
                ResetGarbageFunctor(
                  moodycamel::ConcurrentQueue<std::unique_ptr<Garbage>>* _garbageQueue,
                  std::unique_ptr<Garbage>&& _garb)
                  : garbageQueue(_garbageQueue), garb(std::move(_garb)) {}
                ~ResetGarbageFunctor() override {}
                void call() override { garbageQueue->enqueue(std::move(garb)); }
            };
            frameGraph.getExecutionQueue(recEndHandle)
              ->push(ExecutionPackage(std::make_unique<ResetGarbageFunctor>(
                &garbageQueue, std::move(garbage))));  // TODO create in frame mem
            frameGraph.getExecutionQueue(recEndHandle)
              ->push(ExecutionPackage(ExecutionPackage::MessagePackage{
                ExecutionPackage::Message::RECORD_END, recordFrame, 0, nullptr}));
        }
        {
            std::shared_lock<std::shared_mutex> lock(stopFrameMutex);
            recordFrame++;
        }
    }
    // No node can be waiting for enqueue at this point (or they will never be enqueued)
    frameGraph.getGlobalExecutionQueue()->push(ExecutionPackage(
      ExecutionPackage::MessagePackage{ExecutionPackage::Message::QUIT, 0, 0, nullptr}));
}

bool Engine::execute(Garbage*& garbage, FrameGraph::FrameId& executionFrame,
                     ExecutionPackage&& package) {
    if (std::holds_alternative<ExecutionPackage::MessagePackage>(package.package)) {
        ExecutionPackage::MessagePackage& message =
          std::get<ExecutionPackage::MessagePackage>(package.package);
        switch (message.msg) {
            case ExecutionPackage::Message::RECORD_START: {
                executionFrame = static_cast<FrameGraph::FrameId>(message.value1);
                FrameGraph::NodeHandle executionStartHandle =
                  frameGraph.acquireNode(executeStartNode, executionFrame);
                if (!executionStartHandle) {
                    assert(frameGraph.isStopped());
                    return false;
                }
                garbage = static_cast<Garbage*>(message.valuePtr);
                // std::cout << "Execute start: " << message.value1 << std::endl;
            } break;
            case ExecutionPackage::Message::RECORD_END: {
                assert(executionFrame == static_cast<FrameGraph::FrameId>(message.value1));
                FrameGraph::NodeHandle executionEndHandle =
                  frameGraph.acquireNode(executeEndNode, executionFrame);
                // queued work gets finished
                if (!executionEndHandle) {
                    assert(frameGraph.isStopped());
                    return false;
                }
                // std::cout << "Execute end: " << message.value1 << std::endl;
            } break;
            case ExecutionPackage::Message::PRESENT:
                present(message.value1);
                break;
            case ExecutionPackage::Message::RECURSIVE_END_MARKER:
                break;
            case ExecutionPackage::Message::QUIT:
                return false;
        }
    }
    else if (std::holds_alternative<ExecutionPackage::Functor>(package.package)) {
        ExecutionPackage::Functor& functor = std::get<ExecutionPackage::Functor>(package.package);
        functor();
    }
    else if (std::holds_alternative<std::unique_ptr<ExecutionPackage::CustomFunctor>>(
               package.package)) {
        std::unique_ptr<ExecutionPackage::CustomFunctor>& functor =
          std::get<std::unique_ptr<ExecutionPackage::CustomFunctor>>(package.package);
        functor->call();
    }
    else if (std::holds_alternative<ExecutionPackage::CommandBufferPackage>(package.package)) {
        ExecutionPackage::CommandBufferPackage& cmdBuffer =
          std::get<ExecutionPackage::CommandBufferPackage>(package.package);
        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> signalTimelineSemaphoresMem(
          cmdBuffer.signalTimelineSemaphores.size(), TEMPMEM);
        drv::TimelineSemaphorePtr* signalTimelineSemaphores = signalTimelineSemaphoresMem.get();
        drv::drv_assert(signalTimelineSemaphores != nullptr
                        || cmdBuffer.signalTimelineSemaphores.size() == 0);
        StackMemory::MemoryHandle<uint64_t> signalTimelineSemaphoreValuesMem(
          cmdBuffer.signalTimelineSemaphores.size(), TEMPMEM);
        uint64_t* signalTimelineSemaphoreValues = signalTimelineSemaphoreValuesMem.get();
        drv::drv_assert(signalTimelineSemaphoreValues != nullptr
                        || cmdBuffer.signalTimelineSemaphores.size() == 0);
        for (uint32_t i = 0; i < cmdBuffer.signalTimelineSemaphores.size(); ++i) {
            signalTimelineSemaphores[i] = cmdBuffer.signalTimelineSemaphores[i].semaphore;
            signalTimelineSemaphoreValues[i] = cmdBuffer.signalTimelineSemaphores[i].signalValue;
        }
        drv::ExecutionInfo executionInfo;
        executionInfo.numWaitSemaphores = ;
        executionInfo.waitSemaphores = ;
        executionInfo.waitStages = ;
        if (cmdBuffer.bufferHandle.commandBufferPtr != drv::NULL_HANDLE) {
            executionInfo.numCommandBuffers = 1;
            executionInfo.commandBuffers = &cmdBuffer.bufferHandle.commandBufferPtr;
        }
        else {
            executionInfo.numCommandBuffers = 0;
            executionInfo.commandBuffers = nullptr;
        }
        executionInfo.numSignalSemaphores = cmdBuffer.signalSemaphores.size();
        executionInfo.signalSemaphores = cmdBuffer.signalSemaphores.data();
        executionInfo.numWaitTimelineSemaphores = ;
        executionInfo.waitTimelineSemaphores = ;
        executionInfo.timelineWaitValues = ;
        executionInfo.timelineWaitStages = ;
        executionInfo.numSignalTimelineSemaphores = cmdBuffer.signalTimelineSemaphores.size();
        executionInfo.signalTimelineSemaphores = signalTimelineSemaphores;
        executionInfo.timelineSignalValues = signalTimelineSemaphoreValues;
        drv::execute(cmdBuffer.queue, 1, &executionInfo);
        cmdBuffer.bufferHandle.circulator->startExecution(cmdBuffer.bufferHandle);
        assert(garbage != nullptr && garbage->getFrameId() == executionFrame);
        garbage->resetCommandBuffer(std::move(cmdBuffer.bufferHandle));
    }
    else if (std::holds_alternative<ExecutionPackage::RecursiveQueue>(package.package)) {
        ExecutionPackage::RecursiveQueue& queue =
          std::get<ExecutionPackage::RecursiveQueue>(package.package);
        ExecutionPackage p;
        while (queue.queue->pop(p)) {
            if (std::holds_alternative<ExecutionPackage::MessagePackage>(p.package)) {
                ExecutionPackage::MessagePackage& message =
                  std::get<ExecutionPackage::MessagePackage>(p.package);
                if (message.msg == ExecutionPackage::Message::RECURSIVE_END_MARKER)
                    break;
            }
            if (!execute(garbage, executionFrame, std::move(p)))
                return false;
        }
    }
    return true;
}

void Engine::executeCommandsLoop() {
    std::unique_lock<std::mutex> executionLock(executionMutex);
    Garbage* garbage = nullptr;
    FrameGraph::FrameId executionFrame = 0;
    while (true) {
        ExecutionPackage package;
        ExecutionQueue* executionQueue = frameGraph.getGlobalExecutionQueue();
        executionQueue->waitForPackage();
        while (executionQueue->pop(package))
            if (!execute(garbage, executionFrame, std::move(package)))
                return;
    }
}

void Engine::cleanUpLoop(const volatile std::atomic<FrameGraph::FrameId>* stopFrame) {
    FrameGraph::FrameId cleanUpFrame = 0;
    while (!frameGraph.isStopped() && cleanUpFrame <= *stopFrame) {
        FrameGraph::NodeHandle cleanUpHandle = frameGraph.acquireNode(cleanUpNode, cleanUpFrame);
        if (!cleanUpHandle) {
            assert(frameGraph.isStopped());
            break;
        }
        std::unique_ptr<Garbage> garbage;
        assert(garbageQueue.try_dequeue(garbage));
        assert(garbage && garbage->getFrameId() == cleanUpFrame);
        garbage.reset();  // this does the cleanup
        {
            std::shared_lock<std::shared_mutex> lock(stopFrameMutex);
            cleanUpFrame++;
        }
    }
    {
        // wait for execution queue to finish
        std::unique_lock<std::mutex> executionLock(executionMutex);
        drv::device_wait_idle(device);
        std::unique_ptr<Garbage> garbage;
        while (garbageQueue.try_dequeue(garbage))
            garbage.reset();
    }
}

void Engine::gameLoop() {
    if (simulation == nullptr || renderer == nullptr)
        throw std::runtime_error("Simulation or renderer was not initialized before game loop");
    // entityManager.start();
    // ExecutionQueue executionQueue;
    // RenderState state;
    // state.executionQueue = &executionQueue;

    volatile std::atomic<FrameGraph::FrameId> simulationFrame;
    std::atomic<FrameGraph::FrameId> stopFrame = FrameGraph::INVALID_FRAME;

    std::thread simulationThread(&Engine::simulationLoop, this, &simulationFrame, &stopFrame);
    std::thread recordThread(&Engine::recordCommandsLoop, this, &stopFrame);
    std::thread executeThread(&Engine::executeCommandsLoop, this);
    std::thread cleanUpThread(&Engine::cleanUpLoop, this, &stopFrame);

    try {
        set_thread_name(&simulationThread, "simulation");
        set_thread_name(&recordThread, "record");
        set_thread_name(&executeThread, "execute");
        set_thread_name(&cleanUpThread, "cleanUp");

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
        // frameGraph.stopExecution();
        // }
        {
            std::unique_lock<std::shared_mutex> lock(stopFrameMutex);
            stopFrame = simulationFrame;
        }
        simulationThread.join();
        recordThread.join();
        executeThread.join();
        cleanUpThread.join();
    }
    catch (...) {
        std::cerr << "An exception happend during gameLoop. Waiting for threads to join..."
                  << std::endl;
        frameGraph.stopExecution();
        simulationThread.join();
        recordThread.join();
        executeThread.join();
        cleanUpThread.join();
        throw;
    }
}

Engine::CommandBufferRecorder::CommandBufferRecorder(
  std::unique_lock<std::mutex>&& _queueLock, drv::QueuePtr _queue, FrameGraph::QueueId _queueId,
  FrameGraph* _frameGraph, Engine* _engine, FrameGraph::NodeHandle* _nodeHandle,
  FrameGraph::FrameId _frameId, drv::CommandBufferCirculator::CommandBufferHandle&& _cmdBuffer)
  : queueLock(std::move(_queueLock)),
    queue(_queue),
    queueId(_queueId),
    frameGraph(_frameGraph),
    engine(_engine),
    nodeHandle(_nodeHandle),
    frameId(_frameId),
    cmdBuffer(std::move(_cmdBuffer)) {
    assert(cmdBuffer);
    assert(drv::begin_primary_command_buffer(cmdBuffer.commandBufferPtr, true, false));
}

Engine::CommandBufferRecorder::CommandBufferRecorder(CommandBufferRecorder&& other)
  : queueLock(std::move(other.queueLock)),
    queue(other.queue),
    queueId(other.queueId),
    frameGraph(other.frameGraph),
    engine(other.engine),
    nodeHandle(other.nodeHandle),
    frameId(other.frameId),
    cmdBuffer(std::move(other.cmdBuffer)),
    signalSemaphores(std::move(other.signalSemaphores)),
    signalTimelineSemaphores(std::move(other.signalTimelineSemaphores)) {
    other.engine = nullptr;
}

Engine::CommandBufferRecorder& Engine::CommandBufferRecorder::operator=(
  CommandBufferRecorder&& other) {
    if (&other == this)
        return *this;
    close();
    queueLock = std::move(other.queueLock);
    queue = other.queue;
    queueId = other.queueId;
    frameGraph = other.frameGraph;
    engine = other.engine;
    nodeHandle = other.nodeHandle;
    frameId = other.frameId;
    cmdBuffer = std::move(other.cmdBuffer);
    signalSemaphores = std::move(other.signalSemaphores);
    signalTimelineSemaphores = std::move(other.signalTimelineSemaphores);
    other.engine = nullptr;
    return *this;
}

Engine::CommandBufferRecorder::~CommandBufferRecorder() {
    close();
}

void Engine::CommandBufferRecorder::close() {
    if (engine == nullptr)
        return;
    assert(drv::end_primary_command_buffer(cmdBuffer.commandBufferPtr));
    ExecutionQueue* queue = frameGraph->getExecutionQueue(*nodeHandle);
    queue->push(ExecutionPackage(ExecutionPackage::CommandBufferPackage{
      queue, std::move(cmdBuffer), std::move(signalSemaphores),
      std::move(signalTimelineSemaphores)}));
}

Engine::Garbage::Garbage(FrameGraph::FrameId _frameId) : frameId(_frameId) {
}

Engine::Garbage::Garbage(Garbage&& other)
  : frameId(other.frameId), cmdBuffersToReset(std::move(other.cmdBuffersToReset)) {
}

Engine::Garbage& Engine::Garbage::operator=(Garbage&& other) {
    if (&other == this)
        return *this;
    close();
    frameId = other.frameId;
    cmdBuffersToReset = std::move(other.cmdBuffersToReset);
    return *this;
}

Engine::Garbage::~Garbage() {
    close();
}

void Engine::Garbage::resetCommandBuffer(
  drv::CommandBufferCirculator::CommandBufferHandle&& cmdBuffer) {
    std::unique_lock<std::mutex> lock(mutex);
    cmdBuffersToReset.push_back(std::move(cmdBuffer));
}

void Engine::Garbage::releaseEvent(EventPool::EventHandle&& event) {
    events.push_back(std::move(event));
}

FrameGraph::FrameId Engine::Garbage::getFrameId() const {
    return frameId;
}

void Engine::Garbage::close() noexcept {
    for (auto& cmdBuffer : cmdBuffersToReset)
        cmdBuffer.circulator->finished(std::move(cmdBuffer));
    cmdBuffersToReset.clear();
    events.clear();
}
