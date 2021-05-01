#include "engine.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include <corecontext.h>
#include <util.hpp>

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
            if (data->text)
                std::cout << data->text << std::endl;
            // LOG_DRIVER_API("Driver info: %s", data->text);
            break;
        case drv::CallbackData::Type::NOTE:
            // LOG_DRIVER_API("Driver note: %s", data->text);
            if (data->text)
                std::cout << data->text << std::endl;
            break;
        case drv::CallbackData::Type::WARNING:
            // TODO reword command buffer usage
            LOG_F(WARNING, "Driver warning: %s", data->text);
            // BREAK_POINT;
            // std::cerr << data->text << std::endl;
            break;
        case drv::CallbackData::Type::ERROR:
            LOG_F(ERROR, "Driver error: %s", data->text);
            BREAK_POINT;
            // std::cerr << data->text << std::endl;
            throw std::runtime_error(data->text ? std::string{data->text} : "<0x0>");
        case drv::CallbackData::Type::FATAL:
            LOG_F(ERROR, "Driver warning: %s", data->text);
            BREAK_POINT;
            // std::cerr << data->text << std::endl;
            std::abort();
    }
}

void Engine::Config::writeJson(json& out) const {
    WRITE_OBJECT(screenWidth, out);
    WRITE_OBJECT(screenHeight, out);
    WRITE_OBJECT(stackMemorySizeKb, out);
    WRITE_OBJECT(frameMemorySizeKb, out);
    WRITE_OBJECT(inputBufferSize, out);
    WRITE_OBJECT(title, out);
    WRITE_OBJECT(imagesInSwapchain, out);
    WRITE_OBJECT(maxFramesInExecutionQueue, out);
    WRITE_OBJECT(maxFramesInFlight, out);
    WRITE_OBJECT(driver, out);
    WRITE_OBJECT(logs, out);
    WRITE_OBJECT(trackerConfig, out);
}

void Engine::Config::readJson(const json& in) {
    READ_OBJECT(screenWidth, in);
    READ_OBJECT(screenHeight, in);
    READ_OBJECT(stackMemorySizeKb, in);
    READ_OBJECT(frameMemorySizeKb, in);
    READ_OBJECT(inputBufferSize, in);
    READ_OBJECT(title, in);
    READ_OBJECT(imagesInSwapchain, in);
    READ_OBJECT(maxFramesInExecutionQueue, in);
    READ_OBJECT(maxFramesInFlight, in);
    READ_OBJECT(driver, in);
    READ_OBJECT(logs, in);
    READ_OBJECT_OPT(trackerConfig, in, {});
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
  drv::InstancePtr instance, const drv::DeviceExtensions& deviceExtensions,
  const ShaderBin& shaderBin) {
    drv::PhysicalDevice::SelectionInfo selectInfo;
    selectInfo.instance = instance;
    selectInfo.requirePresent = true;
    selectInfo.extensions = deviceExtensions;
    selectInfo.compare = drv::PhysicalDevice::pick_discere_card;
    selectInfo.commandMasks = {drv::CMD_TYPE_TRANSFER, drv::CMD_TYPE_COMPUTE,
                               drv::CMD_TYPE_GRAPHICS};
    selectInfo.limits = shaderBin.getLimits();
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

Engine::Engine(int argc, char* argv[], const std::string& configFile,
               const std::string& shaderbinFile, ResourceManager::ResourceInfos resource_infos,
               const Args& args)
  : Engine(argc, argv, get_config(configFile), shaderbinFile, std::move(resource_infos), args) {
}

drv::Swapchain::CreateInfo Engine::get_swapchain_create_info(const Config& config,
                                                             drv::QueuePtr present_queue,
                                                             drv::QueuePtr render_queue) {
    drv::Swapchain::CreateInfo ret;
    ret.clipped = true;
    ret.preferredImageCount = static_cast<uint32_t>(config.imagesInSwapchain);
    ret.formatPreferences = {drv::ImageFormat::B8G8R8A8_SRGB};
    ret.preferredPresentModes = {drv::SwapchainCreateInfo::PresentMode::MAILBOX,
                                 drv::SwapchainCreateInfo::PresentMode::IMMEDIATE,
                                 drv::SwapchainCreateInfo::PresentMode::FIFO_RELAXED,
                                 drv::SwapchainCreateInfo::PresentMode::FIFO};
    ret.sharingType = drv::SharingType::CONCURRENT;  // TODO use exclusive
    // TODO usage should come from the game
    ret.usages =
      drv::ImageCreateInfo::COLOR_ATTACHMENT_BIT | drv::ImageCreateInfo::TRANSFER_DST_BIT;
    ret.userQueues = {present_queue, render_queue};
    return ret;
}

Engine::SyncBlock::SyncBlock(drv::LogicalDevicePtr _device, uint32_t maxFramesInFlight) {
    imageAvailableSemaphores.reserve(maxFramesInFlight);
    renderFinishedSemaphores.reserve(maxFramesInFlight);
    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        imageAvailableSemaphores.emplace_back(_device);
        renderFinishedSemaphores.emplace_back(_device);
    }
}

static void log_queue(const char* name, const drv::QueueManager::Queue& queue) {
    LOG_ENGINE("Queue info <%s>: <%p> (G%c C%c T%c) priority:%f index:%d/family:%d", name,
               drv::get_ptr(queue.queue),
               queue.info.commandTypes & drv::CommandTypeBits::CMD_TYPE_GRAPHICS ? '+' : '-',
               queue.info.commandTypes & drv::CommandTypeBits::CMD_TYPE_COMPUTE ? '+' : '-',
               queue.info.commandTypes & drv::CommandTypeBits::CMD_TYPE_TRANSFER ? '+' : '-',
               queue.info.priority, queue.info.queueIndex, queue.info.familyPtr);
}

Engine::Engine(int argc, char* argv[], const Config& cfg, const std::string& shaderbinFile,
               ResourceManager::ResourceInfos resource_infos, const Args& args)
  : config(cfg),
    logger(argc, argv, config.logs),
    coreContext({safe_cast<size_t>(config.stackMemorySizeKb << 10)}),
    shaderBin(shaderbinFile),
    input(safe_cast<size_t>(config.inputBufferSize)),
    driver({get_driver(cfg.driver)}),
    window(&input, &inputManager,
           drv::WindowOptions{static_cast<unsigned int>(cfg.screenWidth),
                              static_cast<unsigned int>(cfg.screenHeight), cfg.title.c_str()}),
    drvInstance(drv::InstanceCreateInfo{cfg.title.c_str(), args.renderdocEnabled,
                                        args.gfxCaptureEnabled, args.apiDumpEnabled}),
    windowIniter(window, drvInstance),
    deviceExtensions(true),
    physicalDevice(get_device_selection_info(drvInstance, deviceExtensions, shaderBin), window),
    commandLaneMgr(physicalDevice, window,
                   {{"main",
                     {{"render", 0.5, drv::CMD_TYPE_GRAPHICS, drv::CMD_TYPE_ALL, false, false},
                      {"present", 0.5, 0, drv::CMD_TYPE_ALL, true, false},
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
    swapchain(physicalDevice, device, window,
              get_swapchain_create_info(config, presentQueue.queue, renderQueue.queue)),
    eventPool(device),
    syncBlock(device, safe_cast<uint32_t>(config.maxFramesInFlight)),
    resourceMgr(std::move(resource_infos)),
    garbageSystem(safe_cast<size_t>(config.frameMemorySizeKb)),
    frameGraph(physicalDevice, device, &garbageSystem, &eventPool, config.trackerConfig),
    runtimeStats(args.runtimeStatsBin.c_str()) {
    json configJson = ISerializable::serialize(config);
    std::stringstream ss;
    ss << configJson;
    LOG_ENGINE("Engine initialized with config: %s", ss.str().c_str());
    // LOG_ENGINE("Build time %s %s", __DATE__, __TIME__);
    log_queue("render", renderQueue);
    log_queue("present", presentQueue);
    log_queue("compute", computeQueue);
    log_queue("DtoH", DtoHQueue);
    log_queue("HtoD", HtoDQueue);
    log_queue("input", inputQueue);
    if (presentQueue.queue != renderQueue.queue
        && drv::can_present(physicalDevice, window,
                            drv::get_queue_family(device, renderQueue.queue)))
        LOG_F(
          WARNING,
          "Present is supported on render queue, but a different queue is selected for presentation");
}

Engine::~Engine() {
    LOG_ENGINE("Engine closed");
}

void Engine::initGame(IRenderer* _renderer, ISimulation* _simulation) {
    RUNTIME_STAT_SCOPE(init);
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
    garbageSystem.resize(cleanUpCpuOffset);

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
    // frameGraph.addDependency(
    //   cleanUpNode, FrameGraph::QueueCpuDependency{presentFrameNode, queueInfos.presentQueue.id, 0});
    frameGraph.addDependency(cleanUpNode, FrameGraph::CpuDependency{executeEndNode, 0});
    frameGraph.addDependency(simStartNode,
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
        // frameGraph.addDependency(presentFrameNode,
        //                          FrameGraph::QueueQueueDependency{renderNode, renderQueueId,
        //                                                           queueInfos.presentQueue.id, 0});
        frameGraph.addDependency(presentFrameNode, FrameGraph::EnqueueDependency{renderNode, 0});
        frameGraph.addDependency(cleanUpNode,
                                 FrameGraph::QueueCpuDependency{renderNode, renderQueueId, 0});
    }

    frameGraph.build();
}

bool Engine::sampleInput(FrameId frameId) {
    FrameGraph::NodeHandle inputHandle = frameGraph.acquireNode(inputSampleNode, frameId);
    if (!inputHandle)
        return false;
    Input::InputEvent event;
    while (input.popEvent(event))
        inputManager.feedInput(std::move(event));
    return true;
}

void Engine::simulationLoop(volatile std::atomic<FrameId>* simulationFrame,
                            const volatile std::atomic<FrameId>* stopFrame) {
    RUNTIME_STAT_SCOPE(simulationLoop);
    loguru::set_thread_name("simulate");
    simulationFrame->store(0);
    while (!frameGraph.isStopped() && *simulationFrame <= *stopFrame) {
        {
            FrameGraph::NodeHandle simStartHandle =
              frameGraph.acquireNode(simStartNode, *simulationFrame);
            if (!simStartHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            garbageSystem.startGarbage(*simulationFrame);
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

void Engine::present(FrameId presentFrame, uint32_t semaphoreIndex) {
    UNUSED(presentFrame);
    drv::PresentInfo info;
    info.semaphoreCount = 1;
    drv::SemaphorePtr semaphore = syncBlock.renderFinishedSemaphores[semaphoreIndex];
    info.waitSemaphores = &semaphore;
    drv::PresentResult result = swapchain.present(presentQueue.queue, info);
    // TODO;  // what's gonna wait on this?
    drv::drv_assert(result != drv::PresentResult::ERROR, "Present error");
    if (result == drv::PresentResult::RECREATE_ADVISED
        || result == drv::PresentResult::RECREATE_REQUIRED) {
        throw std::runtime_error("Implement swapchain recreation and increment swapchainVersion");
        // TODO swapchainVersion ++;
        // state->recreateSwapchain = presentFrame;
    }
}

const Engine::QueueInfo& Engine::getQueues() const {
    return queueInfos;
}

Engine::AcquiredImageData Engine::acquiredSwapchainImage(
  FrameGraph::NodeHandle& acquiringNodeHandle) {
    UNUSED(acquiringNodeHandle);  // TODO latency slop -> acquiringNodeHandle
    bool acquiredSuccess =
      swapchain.acquire(syncBlock.imageAvailableSemaphores[acquireImageSemaphoreId]);
    // ---
    Engine::AcquiredImageData ret;
    if (acquiredSuccess) {
        ret.image = swapchain.getAcquiredImage();
        ret.imageAvailableSemaphore = syncBlock.imageAvailableSemaphores[acquireImageSemaphoreId];
        ret.renderFinishedSemaphore = syncBlock.renderFinishedSemaphores[acquireImageSemaphoreId];
        ret.imageIndex = swapchain.getImageIndex();
        drv::TextureInfo info = drv::get_texture_info(ret.image);
        drv::drv_assert(info.extent.depth == 1);
        ret.extent = {info.extent.width, info.extent.height};
        ret.version = swapchainVersion;
        ret.imageCount = swapchain.getImageCount();
        ret.images = swapchain.getImages();
    }
    else {
        drv::reset_ptr(ret.image);
        drv::reset_ptr(ret.imageAvailableSemaphore);
        drv::reset_ptr(ret.renderFinishedSemaphore);
        ret.imageIndex = 0;
        ret.extent = {0, 0};
        ret.version = INVALID_SWAPCHAIN;
        ret.imageCount = 0;
        ret.images = nullptr;
    }
    acquireImageSemaphoreId =
      (acquireImageSemaphoreId + 1) % syncBlock.imageAvailableSemaphores.size();
    return ret;
}

void Engine::recordCommandsLoop(const volatile std::atomic<FrameId>* stopFrame) {
    RUNTIME_STAT_SCOPE(recordLoop);
    loguru::set_thread_name("record");
    FrameId recordFrame = 0;
    while (!frameGraph.isStopped() && recordFrame <= *stopFrame) {
        // TODO preframe stuff (resize)
        {
            FrameGraph::NodeHandle recStartHandle =
              frameGraph.acquireNode(recordStartNode, recordFrame);
            if (!recStartHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            frameGraph.getExecutionQueue(recStartHandle)
              ->push(ExecutionPackage(ExecutionPackage::MessagePackage{
                ExecutionPackage::Message::RECORD_START, recordFrame, 0, nullptr}));
        }
        const uint32_t semaphoreIndex = acquireImageSemaphoreId;
        renderer->record(frameGraph, recordFrame);
        {
            FrameGraph::NodeHandle presentHandle =
              frameGraph.acquireNode(presentFrameNode, recordFrame);
            if (!presentHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            if (!drv::is_null_ptr(swapchain.getAcquiredImage())) {
                frameGraph.getExecutionQueue(presentHandle)
                  ->push(ExecutionPackage(ExecutionPackage::MessagePackage{
                    ExecutionPackage::Message::PRESENT, recordFrame, semaphoreIndex, nullptr}));
            }
        }
        {
            FrameGraph::NodeHandle recEndHandle =
              frameGraph.acquireNode(recordEndNode, recordFrame);
            if (!recEndHandle) {
                assert(frameGraph.isStopped());
                break;
            }
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

bool Engine::execute(FrameId& executionFrame, ExecutionPackage&& package) {
    if (std::holds_alternative<ExecutionPackage::MessagePackage>(package.package)) {
        ExecutionPackage::MessagePackage& message =
          std::get<ExecutionPackage::MessagePackage>(package.package);
        switch (message.msg) {
            case ExecutionPackage::Message::RECORD_START: {
                executionFrame = static_cast<FrameId>(message.value1);
                FrameGraph::NodeHandle executionStartHandle =
                  frameGraph.acquireNode(executeStartNode, executionFrame);
                if (!executionStartHandle) {
                    assert(frameGraph.isStopped());
                    return false;
                }
                // std::cout << "Execute start: " << message.value1 << std::endl;
            } break;
            case ExecutionPackage::Message::RECORD_END: {
                assert(executionFrame == static_cast<FrameId>(message.value1));
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
                present(message.value1, static_cast<uint32_t>(message.value2));
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
        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> signalTimelineSemaphores(
          cmdBuffer.signalTimelineSemaphores.size(), TEMPMEM);
        StackMemory::MemoryHandle<uint64_t> signalTimelineSemaphoreValues(
          cmdBuffer.signalTimelineSemaphores.size(), TEMPMEM);
        StackMemory::MemoryHandle<drv::SemaphorePtr> waitSemaphores(cmdBuffer.waitSemaphores.size(),
                                                                    TEMPMEM);
        StackMemory::MemoryHandle<drv::PipelineStages::FlagType> waitSemaphoresStages(
          cmdBuffer.waitSemaphores.size(), TEMPMEM);
        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> waitTimelineSemaphores(
          cmdBuffer.waitTimelineSemaphores.size(), TEMPMEM);
        StackMemory::MemoryHandle<drv::PipelineStages::FlagType> waitTimelineSemaphoresStages(
          cmdBuffer.waitTimelineSemaphores.size(), TEMPMEM);
        StackMemory::MemoryHandle<uint64_t> waitTimelineSemaphoresValues(
          cmdBuffer.waitTimelineSemaphores.size(), TEMPMEM);
        for (uint32_t i = 0; i < cmdBuffer.signalTimelineSemaphores.size(); ++i) {
            signalTimelineSemaphores[i] = cmdBuffer.signalTimelineSemaphores[i].semaphore;
            signalTimelineSemaphoreValues[i] = cmdBuffer.signalTimelineSemaphores[i].signalValue;
        }
        for (uint32_t i = 0; i < cmdBuffer.waitSemaphores.size(); ++i) {
            waitSemaphores[i] = cmdBuffer.waitSemaphores[i].semaphore;
            waitSemaphoresStages[i] =
              drv::get_image_usage_stages(cmdBuffer.waitSemaphores[i].imageUsages).stageFlags;
        }
        for (uint32_t i = 0; i < cmdBuffer.waitTimelineSemaphores.size(); ++i) {
            waitTimelineSemaphores[i] = cmdBuffer.waitTimelineSemaphores[i].semaphore;
            waitTimelineSemaphoresValues[i] = cmdBuffer.waitTimelineSemaphores[i].waitValue;
            waitTimelineSemaphoresStages[i] =
              drv::get_image_usage_stages(cmdBuffer.waitTimelineSemaphores[i].imageUsages)
                .stageFlags;
        }
        drv::ExecutionInfo executionInfo;
        executionInfo.numWaitSemaphores =
          static_cast<unsigned int>(cmdBuffer.waitSemaphores.size());
        executionInfo.waitSemaphores = waitSemaphores;
        executionInfo.waitStages = waitSemaphoresStages;
        if (!drv::is_null_ptr(cmdBuffer.cmdBufferPtr)) {
            executionInfo.numCommandBuffers = 1;
            executionInfo.commandBuffers = &cmdBuffer.cmdBufferPtr;
        }
        else {
            executionInfo.numCommandBuffers = 0;
            executionInfo.commandBuffers = nullptr;
        }
        executionInfo.numSignalSemaphores =
          static_cast<unsigned int>(cmdBuffer.signalSemaphores.size());
        executionInfo.signalSemaphores = cmdBuffer.signalSemaphores.data();
        executionInfo.numWaitTimelineSemaphores =
          static_cast<unsigned int>(cmdBuffer.waitTimelineSemaphores.size());
        executionInfo.waitTimelineSemaphores = waitTimelineSemaphores;
        executionInfo.timelineWaitValues = waitTimelineSemaphoresValues;
        executionInfo.timelineWaitStages = waitTimelineSemaphoresStages;
        executionInfo.numSignalTimelineSemaphores =
          static_cast<unsigned int>(cmdBuffer.signalTimelineSemaphores.size());
        executionInfo.signalTimelineSemaphores = signalTimelineSemaphores;
        executionInfo.timelineSignalValues = signalTimelineSemaphoreValues;
        drv::execute(cmdBuffer.queue, 1, &executionInfo);
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
            if (!execute(executionFrame, std::move(p)))
                return false;
        }
    }
    return true;
}

void Engine::executeCommandsLoop() {
    RUNTIME_STAT_SCOPE(executionLoop);
    loguru::set_thread_name("execution");
    std::unique_lock<std::mutex> executionLock(executionMutex);
    FrameId executionFrame = 0;
    while (true) {
        ExecutionPackage package;
        ExecutionQueue* executionQueue = frameGraph.getGlobalExecutionQueue();
        executionQueue->waitForPackage();
        while (executionQueue->pop(package))
            if (!execute(executionFrame, std::move(package)))
                return;
    }
}

void Engine::cleanUpLoop(const volatile std::atomic<FrameId>* stopFrame) {
    RUNTIME_STAT_SCOPE(cleanupLoop);
    loguru::set_thread_name("clean up");
    FrameId cleanUpFrame = 0;
    while (!frameGraph.isStopped() && cleanUpFrame <= *stopFrame) {
        FrameGraph::NodeHandle cleanUpHandle = frameGraph.acquireNode(cleanUpNode, cleanUpFrame);
        if (!cleanUpHandle) {
            assert(frameGraph.isStopped());
            break;
        }
        garbageSystem.releaseGarbage(cleanUpFrame);
        {
            std::shared_lock<std::shared_mutex> lock(stopFrameMutex);
            cleanUpFrame++;
        }
    }
    {
        // wait for execution queue to finish
        std::unique_lock<std::mutex> executionLock(executionMutex);
        drv::device_wait_idle(device);
        garbageSystem.releaseAll();
    }
}

void Engine::gameLoop() {
    RUNTIME_STAT_SCOPE(gameLoop);
    if (simulation == nullptr || renderer == nullptr)
        throw std::runtime_error("Simulation or renderer was not initialized before game loop");
    // entityManager.start();
    // ExecutionQueue executionQueue;
    // RenderState state;
    // state.executionQueue = &executionQueue;

    volatile std::atomic<FrameId> simulationFrame;
    std::atomic<FrameId> stopFrame = INVALID_FRAME;

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
