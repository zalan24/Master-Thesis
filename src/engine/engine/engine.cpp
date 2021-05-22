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

#include <namethreads.h>

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
            break;
        case drv::CallbackData::Type::ERROR:
            LOG_F(ERROR, "Driver error: %s", data->text);
            BREAK_POINT;
            throw std::runtime_error(data->text ? std::string{data->text} : "<0x0>");
        case drv::CallbackData::Type::FATAL:
            LOG_F(ERROR, "Driver warning: %s", data->text);
            BREAK_POINT;
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

Engine::Engine(int argc, char* argv[], const Config& cfg,
               const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
               ResourceManager::ResourceInfos resource_infos, const Args& args)
  : config(cfg),
    logger(argc, argv, config.logs),
    coreContext({safe_cast<size_t>(config.stackMemorySizeKb << 10)}),
    shaderBin(shaderbinFile),
    input(safe_cast<size_t>(config.inputBufferSize)),
    driver(trackingConfig, {get_driver(cfg.driver)}),
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
    syncBlock(device, safe_cast<uint32_t>(config.maxFramesInFlight)),  // TODO why just 2?
    resourceMgr(std::move(resource_infos)),
    garbageSystem(safe_cast<size_t>(config.frameMemorySizeKb) << 10),
    // maxFramesInFlight + 1 for readback stage
    frameGraph(physicalDevice, device, &garbageSystem, &eventPool, config.trackerConfig,
               config.maxFramesInExecutionQueue, config.maxFramesInFlight + 1),
    runtimeStats(args.runtimeStatsBin.c_str()) {
    json configJson = ISerializable::serialize(config);
    std::stringstream ss;
    ss << configJson;
    LOG_ENGINE("Engine initialized with config: %s", ss.str().c_str());
    LOG_ENGINE("Build time %s %s", __DATE__, __TIME__);
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

    queueInfos.computeQueue = {computeQueue.queue, frameGraph.registerQueue(computeQueue.queue)};
    queueInfos.DtoHQueue = {DtoHQueue.queue, frameGraph.registerQueue(DtoHQueue.queue)};
    queueInfos.HtoDQueue = {HtoDQueue.queue, frameGraph.registerQueue(HtoDQueue.queue)};
    queueInfos.inputQueue = {inputQueue.queue, frameGraph.registerQueue(inputQueue.queue)};
    queueInfos.presentQueue = {presentQueue.queue, frameGraph.registerQueue(presentQueue.queue)};
    queueInfos.renderQueue = {renderQueue.queue, frameGraph.registerQueue(renderQueue.queue)};

    inputSampleNode =
      frameGraph.addNode(FrameGraph::Node("sample_input", FrameGraph::SIMULATION_STAGE));
    presentFrameNode = frameGraph.addNode(
      FrameGraph::Node("presentFrame", FrameGraph::RECORD_STAGE | FrameGraph::EXECUTION_STAGE));

    drv::drv_assert(config.maxFramesInExecutionQueue >= 1,
                    "maxFramesInExecutionQueue must be at least 1");
    drv::drv_assert(config.maxFramesInFlight >= 2, "maxFramesInFlight must be at least 2");
    drv::drv_assert(config.maxFramesInFlight >= config.maxFramesInExecutionQueue,
                    "maxFramesInFlight must be at least the value of maxFramesInExecutionQueue");
    const uint32_t presentDepOffset = static_cast<uint32_t>(config.maxFramesInFlight - 1);
    garbageSystem.resize(frameGraph.getMaxFramesInFlight());

    // TODO this can't really auto wait now, because there is now command buffer that waits on the present...
    frameGraph.addDependency(
      presentFrameNode, FrameGraph::QueueCpuDependency{presentFrameNode, queueInfos.presentQueue.id,
                                                       FrameGraph::RECORD_STAGE, presentDepOffset});
}

void Engine::buildFrameGraph(FrameGraph::NodeId presentDepNode, FrameGraph::QueueId depQueueId) {
    frameGraph.addDependency(presentFrameNode, FrameGraph::EnqueueDependency{presentDepNode, 0});
    // TODO this should be done automatically
    frameGraph.addDependency(
      frameGraph.getStageEndNode(FrameGraph::READBACK_STAGE),
      FrameGraph::QueueCpuDependency{presentDepNode, depQueueId, FrameGraph::READBACK_STAGE, 0});

    frameGraph.build();
}

Engine::~Engine() {
    LOG_ENGINE("Engine closed");
}

bool Engine::sampleInput(FrameId frameId) {
    FrameGraph::NodeHandle inputHandle =
      frameGraph.acquireNode(inputSampleNode, FrameGraph::SIMULATION_STAGE, frameId);
    if (!inputHandle)
        return false;
    Input::InputEvent event;
    while (input.popEvent(event))
        inputManager.feedInput(std::move(event));
    return true;
}

void Engine::simulationLoop() {
    RUNTIME_STAT_SCOPE(simulationLoop);
    loguru::set_thread_name("simulate");
    FrameId simulationFrame = 0;
    while (!frameGraph.isStopped()) {
        mainKernelCv.notify_one();
        if (auto startNode =
              frameGraph.acquireNode(frameGraph.getStageStartNode(FrameGraph::SIMULATION_STAGE),
                                     FrameGraph::SIMULATION_STAGE, simulationFrame);
            startNode) {
            garbageSystem.startGarbage(simulationFrame);
        }
        else
            break;
        if (!sampleInput(simulationFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        simulate(simulationFrame);
        if (!frameGraph.endStage(FrameGraph::SIMULATION_STAGE, simulationFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        simulationFrame++;
    }
}

void Engine::recreateSwapchain() {
    std::unique_lock<std::mutex> lock(swapchainMutex);
    // images need to keep the same memory address
    drv::Swapchain::OldSwapchinData oldSwapchain = swapchain.recreate(physicalDevice, window);
    garbageSystem.useGarbage([&](Garbage* trashBin) {
        trashBin->releaseTrash(
          std::make_unique<Garbage::TrashData<drv::Swapchain::OldSwapchinData>>(
            std::move(oldSwapchain)));
    });
    swapchainVersion++;
}

void Engine::present(drv::SwapchainPtr swapchainPtr, FrameId frameId, uint32_t imageIndex,
                     uint32_t semaphoreIndex) {
    if (drv::is_null_ptr(swapchainPtr))
        return;
    if (firstPresentableFrame > frameId)
        LOG_F(WARNING,
              "Presenting to an invalid swapchain. Wait on the semaphore to fix this warning");
    // if (firstPresentableFrame <= frameId) {
    drv::PresentInfo info;
    info.semaphoreCount = 1;
    drv::SemaphorePtr semaphore = syncBlock.renderFinishedSemaphores[semaphoreIndex];
    info.waitSemaphores = &semaphore;
    std::unique_lock<std::mutex> lock(swapchainMutex);
    drv::PresentResult result =
      swapchain.present(presentQueue.queue, swapchainPtr, info, imageIndex);
    // TODO;  // what's gonna wait on this?
    drv::drv_assert(result != drv::PresentResult::ERROR, "Present error");
    // }
    // else {
    // }
}

const Engine::QueueInfo& Engine::getQueues() const {
    return queueInfos;
}

Engine::AcquiredImageData Engine::acquiredSwapchainImage(
  FrameGraph::NodeHandle& acquiringNodeHandle) {
    Engine::AcquiredImageData ret;
    drv::Extent2D windowExtent = window->getResolution();
    if (windowExtent.width != 0 && windowExtent.height != 0
        && windowExtent == swapchain.getCurrentEXtent()) {
        uint32_t imageIndex = drv::Swapchain::INVALID_INDEX;
        std::unique_lock<std::mutex> lock(swapchainMutex);
        // TODO latency slop -> acquiringNodeHandle
        drv::AcquireResult result = swapchain.acquire(
          imageIndex, syncBlock.imageAvailableSemaphores[acquireImageSemaphoreId]);
        // ---
        switch (result) {
            case drv::AcquireResult::ERROR:
                drv::drv_assert(false, "Could not acquire swapchain image");
                break;
            case drv::AcquireResult::ERROR_RECREATE_REQUIRED:
                swapchainState = SwapchainState::INVALID;
                break;
            case drv::AcquireResult::TIME_OUT:
                break;
            case drv::AcquireResult::SUCCESS_RECREATE_ADVISED:
            case drv::AcquireResult::SUCCESS_NOT_READY:
            case drv::AcquireResult::SUCCESS:
                if (imageIndex != drv::Swapchain::INVALID_INDEX) {
                    ret.image = swapchain.getAcquiredImage(imageIndex);
                    ret.swapchain = swapchain;
                    ret.semaphoreIndex = acquireImageSemaphoreId;
                    ret.imageAvailableSemaphore =
                      syncBlock.imageAvailableSemaphores[acquireImageSemaphoreId];
                    ret.renderFinishedSemaphore =
                      syncBlock.renderFinishedSemaphores[acquireImageSemaphoreId];
                    ret.imageIndex = imageIndex;
                    drv::TextureInfo info = drv::get_texture_info(ret.image);
                    drv::drv_assert(info.extent.depth == 1);
                    ret.extent = {info.extent.width, info.extent.height};
                    ret.version = swapchainVersion;
                    ret.imageCount = swapchain.getImageCount();
                    ret.images = swapchain.getImages();
                    acquireImageSemaphoreId =
                      (acquireImageSemaphoreId + 1) % syncBlock.imageAvailableSemaphores.size();
                }
                break;
        }
        if (result == drv::AcquireResult::SUCCESS_RECREATE_ADVISED)
            swapchainState = SwapchainState::OKAY;
        if (result == drv::AcquireResult::SUCCESS)
            swapchainState = SwapchainState::OK;
    }
    if (drv::is_null_ptr(ret.swapchain))
        firstPresentableFrame = acquiringNodeHandle.getFrameId() + 1;
    return ret;
}

void Engine::beforeDrawLoop() {
    RUNTIME_STAT_SCOPE(beforeDrawLoop);
    loguru::set_thread_name("beforeDraw");
    FrameId beforeDrawFrame = 0;
    while (!frameGraph.isStopped()) {
        if (!frameGraph.startStage(FrameGraph::BEFORE_DRAW_STAGE, beforeDrawFrame))
            break;
        beforeDraw(beforeDrawFrame);
        if (!frameGraph.endStage(FrameGraph::BEFORE_DRAW_STAGE, beforeDrawFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        beforeDrawFrame++;
    }
}

void Engine::recordCommandsLoop() {
    RUNTIME_STAT_SCOPE(recordLoop);
    loguru::set_thread_name("record");
    FrameId recordFrame = 0;
    while (!frameGraph.isStopped()) {
        mainKernelCv.notify_one();
        if (!frameGraph.startStage(FrameGraph::RECORD_STAGE, recordFrame))
            break;
        AcquiredImageData swapchainData = record(recordFrame);
        {
            FrameGraph::NodeHandle presentHandle =
              frameGraph.acquireNode(presentFrameNode, FrameGraph::RECORD_STAGE, recordFrame);
            if (!presentHandle) {
                assert(frameGraph.isStopped());
                break;
            }
            if (!drv::is_null_ptr(swapchainData.image)) {
                frameGraph.getExecutionQueue(presentHandle)
                  ->push(ExecutionPackage(ExecutionPackage::PresentPackage{
                    recordFrame, swapchainData.imageIndex, swapchainData.semaphoreIndex,
                    swapchainData.swapchain}));
            }
        }
        if (!frameGraph.endStage(FrameGraph::RECORD_STAGE, recordFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        recordFrame++;
    }
    // No node can be waiting for enqueue at this point (or they will never be enqueued)
    frameGraph.getGlobalExecutionQueue()->push(ExecutionPackage(
      ExecutionPackage::MessagePackage{ExecutionPackage::Message::QUIT, 0, 0, nullptr}));
}

bool Engine::execute(ExecutionPackage&& package) {
    if (std::holds_alternative<ExecutionPackage::MessagePackage>(package.package)) {
        ExecutionPackage::MessagePackage& message =
          std::get<ExecutionPackage::MessagePackage>(package.package);
        switch (message.msg) {
            case ExecutionPackage::Message::FRAMEGRAPH_NODE_MARKER: {
                FrameGraph::NodeId nodeId = static_cast<FrameGraph::NodeId>(message.value1);
                FrameId frame = static_cast<FrameId>(message.value2);
                frameGraph.executionFinished(nodeId, frame);
            } break;
            case ExecutionPackage::Message::RECURSIVE_END_MARKER:
                break;
            case ExecutionPackage::Message::QUIT:
                return false;
        }
    }
    else if (std::holds_alternative<ExecutionPackage::PresentPackage>(package.package)) {
        ExecutionPackage::PresentPackage& p =
          std::get<ExecutionPackage::PresentPackage>(package.package);
        present(p.swapichain, p.frame, p.imageIndex, p.semaphoreId);
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
        drv::CommandBufferPtr commandBuffers[2];
        uint32_t numCommandBuffers = 0;

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
        {
            drv::StateCorrectionData correctionData;
            if (!drv::validate_and_apply_state_transitions(
                  correctionData, uint32_t(cmdBuffer.cmdBufferData.imageStates.size()),
                  cmdBuffer.cmdBufferData.imageStates.data())) {
                if (cmdBuffer.cmdBufferData.stateValidation) {
                    LOG_F(ERROR, "Some resources are not in the expected state");
                    BREAK_POINT;
                }
                OneTimeCmdBuffer<const drv::StateCorrectionData*> correctionCmdBuffer(
                  physicalDevice, device, cmdBuffer.queue, getCommandBufferBank(),
                  getGarbageSystem(),
                  [](const drv::StateCorrectionData* const& data,
                     drv::DrvCmdBufferRecorder* recorder) { recorder->corrigate(*data); });
                commandBuffers[numCommandBuffers++] =
                  correctionCmdBuffer.use(&correctionData).cmdBufferPtr;
            }
            if (!drv::is_null_ptr(cmdBuffer.cmdBufferData.cmdBufferPtr))
                commandBuffers[numCommandBuffers++] = cmdBuffer.cmdBufferData.cmdBufferPtr;
        }
        executionInfo.numCommandBuffers = numCommandBuffers;
        executionInfo.commandBuffers = commandBuffers;
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
            if (!execute(std::move(p)))
                return false;
        }
    }
    return true;
}

void Engine::executeCommandsLoop() {
    RUNTIME_STAT_SCOPE(executionLoop);
    loguru::set_thread_name("execution");
    std::unique_lock<std::mutex> executionLock(executionMutex);
    while (true) {
        ExecutionPackage package;
        ExecutionQueue* executionQueue = frameGraph.getGlobalExecutionQueue();
        executionQueue->waitForPackage();
        while (executionQueue->pop(package))
            if (!execute(std::move(package)))
                return;
    }
}

void Engine::readbackLoop() {
    RUNTIME_STAT_SCOPE(readbackLoop);
    loguru::set_thread_name("readback");
    FrameId readbackFrame = 0;
    while (!frameGraph.isStopped()) {
        if (!frameGraph.startStage(FrameGraph::READBACK_STAGE, readbackFrame))
            break;
        readback(readbackFrame);
        if (auto handle =
              frameGraph.acquireNode(frameGraph.getStageEndNode(FrameGraph::READBACK_STAGE),
                                     FrameGraph::READBACK_STAGE, readbackFrame);
            handle) {
            garbageSystem.releaseGarbage(readbackFrame);
        }
        else {
            assert(frameGraph.isStopped());
            break;
        }
        readbackFrame++;
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
    // entityManager.start();
    // ExecutionQueue executionQueue;
    // RenderState state;
    // state.executionQueue = &executionQueue;

    std::thread simulationThread(&Engine::simulationLoop, this);
    std::thread beforeDrawThread(&Engine::beforeDrawLoop, this);
    std::thread recordThread(&Engine::recordCommandsLoop, this);
    std::thread executeThread(&Engine::executeCommandsLoop, this);
    std::thread readbackThread(&Engine::readbackLoop, this);

    try {
        set_thread_name(&simulationThread, "simulation");
        set_thread_name(&beforeDrawThread, "beforeDraw");
        set_thread_name(&recordThread, "record");
        set_thread_name(&executeThread, "execute");
        set_thread_name(&readbackThread, "readback");

        IWindow* w = window;
        while (!w->shouldClose()) {
            mainLoopKernel();
        }
        frameGraph.stopExecution(false);
        simulationThread.join();
        beforeDrawThread.join();
        recordThread.join();
        executeThread.join();
        readbackThread.join();
    }
    catch (const std::exception& e) {
        LOG_F(ERROR, "An exception happend during gameLoop. Waiting for threads to join: <%s>",
              e.what());
        BREAK_POINT;
        frameGraph.stopExecution(true);
        simulationThread.join();
        beforeDrawThread.join();
        recordThread.join();
        executeThread.join();
        readbackThread.join();
        throw;
    }
    catch (...) {
        LOG_F(ERROR, "An exception happend during gameLoop. Waiting for threads to join");
        BREAK_POINT;
        frameGraph.stopExecution(true);
        simulationThread.join();
        beforeDrawThread.join();
        recordThread.join();
        executeThread.join();
        readbackThread.join();
        throw;
    }
}

void Engine::mainLoopKernel() {
    std::unique_lock<std::mutex> lock(mainKernelMutex);
    mainKernelCv.wait_for(lock, std::chrono::milliseconds(4));
    static_cast<IWindow*>(window)->pollEvents();

    if (garbageSystem.getStartedFrame() != INVALID_FRAME) {
        drv::Extent2D windowExtent = window->getResolution();
        if (windowExtent.width != 0 && windowExtent.height != 0) {
            if (windowExtent != swapchain.getCurrentEXtent()
                || (swapchainState != SwapchainState::OK
                    && swapchainState != SwapchainState::UNKNOWN)) {
                recreateSwapchain();
                swapchainState = SwapchainState::UNKNOWN;
            }
        }
    }
}
