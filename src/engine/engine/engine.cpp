#include "engine.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include <features.h>

#include <corecontext.h>
#include <util.hpp>

#include <drv.h>
#include <drverror.h>
#include <drvwindow.h>

#include <namethreads.h>

#include "execution_queue.h"
#include "imagestager.h"

bool EngineMouseInputListener::processMouseButton(const Input::MouseButtenEvent& event) {
    if (event.buttonId == 0) {
        if (event.type == event.PRESS)
            clicking = true;
        else
            clicking = false;
    }
    return false;
}

bool EngineMouseInputListener::processMouseMove(const Input::MouseMoveEvent& event) {
    mX = event.relX;
    mY = event.relY;
    return false;
}

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

#if ENABLE_DYNAMIC_ALLOCATION_DEBUG
// TODO add to report file
void* operator new(size_t size) {
    void* p = std::malloc(size);
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}
#endif

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

drv::Swapchain::CreateInfo Engine::get_swapchain_create_info(const EngineConfig& config,
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

uint32_t Engine::getMaxFramesInFlight() const {
    return frameGraph.getMaxFramesInFlight();
}

Engine::Engine(int argc, char* argv[], const EngineConfig& cfg,
               const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
               const Resources& _resources, Args _args)
  : config(cfg),
    resourceFolders(_resources),
    launchArgs(std::move(_args)),
    workLoadFile("workLoad.json"),
    logger(argc, argv, config.logs),
    coreContext(std::make_unique<CoreContext>(
      CoreContext::Config{safe_cast<size_t>(config.stackMemorySizeKb << 10)})),
    garbageSystem(safe_cast<size_t>(config.frameMemorySizeKb) << 10),
    shaderBin(shaderbinFile),
    input(safe_cast<size_t>(config.inputBufferSize)),
    driver(trackingConfig, {get_driver(cfg.driver)}),
    window(&input, &inputManager,
           drv::WindowOptions{static_cast<unsigned int>(cfg.screenWidth),
                              static_cast<unsigned int>(cfg.screenHeight), cfg.title.c_str()}),
    drvInstance(drv::InstanceCreateInfo{cfg.title.c_str(), launchArgs.renderdocEnabled,
                                        launchArgs.gfxCaptureEnabled, launchArgs.apiDumpEnabled}),
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
    semaphorePool(device, config.maxFramesInFlight),
    swapchain(physicalDevice, device, window,
              get_swapchain_create_info(config, presentQueue.queue, renderQueue.queue)),
    eventPool(device),
    syncBlock(device, safe_cast<uint32_t>(config.maxFramesInFlight)),  // TODO why just 2?
    // maxFramesInFlight + 1 for readback stage
    frameGraph(physicalDevice, device, &garbageSystem, &resourceLocker, &eventPool, &semaphorePool,
               trackingConfig, config.maxFramesInExecutionQueue, config.maxFramesInFlight + 1),
    runtimeStats(!launchArgs.clearRuntimeStats, launchArgs.runtimeStatsPersistanceBin,
                 launchArgs.runtimeStatsGameExportsBin, launchArgs.runtimeStatsCacheBin),
    entityManager(physicalDevice, device, &frameGraph, resourceFolders.textures) {
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
    mainRecordNode = frameGraph.addNode(
      FrameGraph::Node("mainRecord", FrameGraph::RECORD_STAGE | FrameGraph::EXECUTION_STAGE));

    drv::drv_assert(config.maxFramesInExecutionQueue >= 1,
                    "maxFramesInExecutionQueue must be at least 1");
    drv::drv_assert(config.maxFramesInFlight >= 2, "maxFramesInFlight must be at least 2");
    drv::drv_assert(config.maxFramesInFlight >= config.maxFramesInExecutionQueue,
                    "maxFramesInFlight must be at least the value of maxFramesInExecutionQueue");
    const uint32_t presentDepOffset = static_cast<uint32_t>(config.maxFramesInFlight - 1);
    garbageSystem.resize(frameGraph.getMaxFramesInFlight());

    frameGraph.addAllGpuCompleteDependency(presentFrameNode, FrameGraph::RECORD_STAGE,
                                           presentDepOffset);
    inputManager.registerListener(&mouseListener, 100);
}

void Engine::buildFrameGraph() {
    frameGraph.addDependency(presentFrameNode, FrameGraph::EnqueueDependency{mainRecordNode, 0});

    drv::drv_assert(renderEntitySystem.flag != 0, "Render entity system was not registered ");
    drv::drv_assert(physicsEntitySystem.flag != 0, "Physics entity system was not registered ");
    drv::drv_assert(cursorEntitySystem.flag != 0, "Cursor entity system was not registered ");
    drv::drv_assert(latencyFlashEntitySystem.flag != 0,
                    "Latency flash entity system was not registered ");
    drv::drv_assert(cameraEntitySystem.flag != 0, "Camera flash entity system was not registered ");

    entityManager.addEntityTemplate("static",
                                    {physicsEntitySystem.flag | renderEntitySystem.flag, 0});
    entityManager.addEntityTemplate("cursor",
                                    {cursorEntitySystem.flag | renderEntitySystem.flag, 0});
    entityManager.addEntityTemplate("latencyFlash",
                                    {latencyFlashEntitySystem.flag | renderEntitySystem.flag, 0});
    entityManager.addEntityTemplate("camera",
                                    {physicsEntitySystem.flag | cameraEntitySystem.flag, 0});

    frameGraph.addDependency(
      renderEntitySystem.nodeId,
      FrameGraph::CpuDependency{latencyFlashEntitySystem.nodeId, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::BEFORE_DRAW_STAGE, 0});
    frameGraph.addDependency(
      renderEntitySystem.nodeId,
      FrameGraph::CpuDependency{cursorEntitySystem.nodeId, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::BEFORE_DRAW_STAGE, 0});
    frameGraph.addDependency(
      renderEntitySystem.nodeId,
      FrameGraph::CpuDependency{cameraEntitySystem.nodeId, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::BEFORE_DRAW_STAGE, 0});
    frameGraph.addDependency(
      cameraEntitySystem.nodeId,
      FrameGraph::CpuDependency{inputSampleNode, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::SIMULATION_STAGE, 0});
    frameGraph.addDependency(
      cursorEntitySystem.nodeId,
      FrameGraph::CpuDependency{inputSampleNode, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::SIMULATION_STAGE, 0});
    frameGraph.addDependency(
      latencyFlashEntitySystem.nodeId,
      FrameGraph::CpuDependency{inputSampleNode, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::SIMULATION_STAGE, 0});
    frameGraph.addDependency(
      mainRecordNode,
      FrameGraph::CpuDependency{renderEntitySystem.nodeId, FrameGraph::BEFORE_DRAW_STAGE,
                                FrameGraph::RECORD_STAGE, 0});
    frameGraph.addDependency(renderEntitySystem.nodeId,
                             FrameGraph::CpuDependency{mainRecordNode, FrameGraph::RECORD_STAGE,
                                                       FrameGraph::BEFORE_DRAW_STAGE, 1});

    entityManager.initFrameGraph();
    frameGraph.build();

    if (!fs::exists(fs::path{workLoadFile})) {
        ArtificialFrameGraphWorkLoad workLoad = frameGraph.getWorkLoad();
        drv::drv_assert(workLoad.exportToFile(fs::path{workLoadFile}),
                        "Could not generate work load file");
        LOG_ENGINE("Work load file generated: %s", workLoadFile.c_str());
    }
    else {
        ArtificialFrameGraphWorkLoad workLoad;
        drv::drv_assert(workLoad.importFromFile(fs::path{workLoadFile}),
                        "Could not import artificial work load");
        frameGraph.setWorkLoad(workLoad);
    }
    workLoadFileModificationDate = fs::last_write_time(fs::path{workLoadFile});

    entityManager.importFromFile(fs::path{launchArgs.sceneToLoad});
}

void Engine::esCamera(EntityManager*, Engine* engine, FrameGraph::NodeHandle*, FrameGraph::Stage,
                      const EntityManager::EntitySystemParams&, Entity* entity) {
    drv::Extent2D extent = engine->window->getResolution();
    entity->extra["ratio"] = float(extent.height) / float(extent.width);
}

void Engine::esPhysics(EntityManager*, Engine*, FrameGraph::NodeHandle*, FrameGraph::Stage,
                       const EntityManager::EntitySystemParams& params, Entity* entity) {
    entity->position += entity->speed * params.dt;
}

void Engine::esBeforeDraw(EntityManager* entityManager, Engine* engine, FrameGraph::NodeHandle*,
                          FrameGraph::Stage, const EntityManager::EntitySystemParams&,
                          Entity* entity) {
    if (entity->hidden)
        return;
    EntityRenderData data;
    glm::vec2 camPos;
    glm::vec2 camSize;
    {
        const Entity* camEntity = entityManager->getById(entityManager->getByName("camera"));
        std::shared_lock<std::shared_mutex> camLock(camEntity->mutex);
        camPos = camEntity->position;
        camSize = camEntity->scale;
        camSize.y *= camEntity->extra.find("ratio")->second;
    }
    data.z = entity->zPos;
    data.textureId = entity->textureId;
    Entity::EntityId parentId = entity->parent;
    glm::vec2 pos = entity->position;
    glm::vec2 scale = entity->scale;
    while (parentId != Entity::INVALID_ENTITY) {
        const Entity* parentEntity = entityManager->getById(parentId);
        pos = parentEntity->scale * pos + parentEntity->position;
        scale = scale * parentEntity->scale;
        parentId = parentEntity->parent;
    }
    data.relBottomLeft = (pos - scale - camPos);
    data.relBottomLeft.x /= camSize.x;
    data.relBottomLeft.y /= camSize.y;
    data.relTopRight = (pos + scale - camPos);
    data.relTopRight.x /= camSize.x;
    data.relTopRight.y /= camSize.y;
    if (data.relBottomLeft.x < 1 && data.relTopRight.x > -1 && data.relBottomLeft.y < 1
        && data.relTopRight.y > -1)
        engine->entitiesToDraw.push_back(std::move(data));
}

void Engine::esLatencyFlash(EntityManager*, Engine* engine, FrameGraph::NodeHandle*,
                            FrameGraph::Stage, const EntityManager::EntitySystemParams& params,
                            Entity* entity) {
    if (engine->mouseListener.isClicking())
        entity->extra["visibleTime"] = params.uptime + 0.3f;
    entity->hidden = !(params.uptime < entity->extra["visibleTime"]);
}

void Engine::esCursor(EntityManager* entityManager, Engine* engine, FrameGraph::NodeHandle*,
                      FrameGraph::Stage, const EntityManager::EntitySystemParams&, Entity* entity) {
    const Entity* camEntity = entityManager->getById(entityManager->getByName("camera"));
    std::shared_lock<std::shared_mutex> camLock(camEntity->mutex);
    entity->position = engine->mouseListener.getMousePos() * 2.f - 1.f;
    if (auto itr = camEntity->extra.find("ratio"); itr != camEntity->extra.end())
        entity->position.y *= itr->second;
    // EntityRenderData data;
    // glm::vec2 camPos = {0, 0};
    // glm::vec2 camSize = {10, 10};
    // data.relBottomLeft = (entity->position - entity->scale - camPos);
    // data.relBottomLeft.x /= camSize.x;
    // data.relBottomLeft.y /= camSize.y;
    // data.relTopRight = (entity->position + entity->scale - camPos);
    // data.relTopRight.x /= camSize.x;
    // data.relTopRight.y /= camSize.y;
    // data.z = entity->zPos;
    // data.textureId = entity->textureId;
    // if (data.relBottomLeft.x < 1 && data.relTopRight.x > -1 && data.relBottomLeft.y < 1
    //     && data.relTopRight.y > -1)
    //     engine->entitiesToDraw.push_back(std::move(data));
}

void Engine::initPhysicsEntitySystem() {
    physicsEntitySystem = entityManager.addEntitySystem(
      "entityPhysics", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esPhysics);
}

void Engine::initRenderEntitySystem() {
    renderEntitySystem = entityManager.addEntitySystem(
      "entityRender", FrameGraph::BEFORE_DRAW_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, true}, esBeforeDraw);
}

void Engine::initCursorEntitySystem() {
    cursorEntitySystem = entityManager.addEntitySystem(
      "cursorES", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esCursor);
    latencyFlashEntitySystem = entityManager.addEntitySystem(
      "latencyFlashES", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esLatencyFlash);
    cameraEntitySystem = entityManager.addEntitySystem(
      "cameraES", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esCamera);
}

Engine::~Engine() {
    garbageSystem.releaseAll();
    inputManager.unregisterListener(&mouseListener);
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

        auto modificationDate = fs::last_write_time(fs::path{workLoadFile});
        if (workLoadFileModificationDate != modificationDate) {
            workLoadFileModificationDate = modificationDate;
            ArtificialFrameGraphWorkLoad workLoad;
            try {
                if (workLoad.importFromFile(fs::path{workLoadFile})) {
                    frameGraph.setWorkLoad(workLoad);
                    LOG_ENGINE("Work load file reloaded");
                }
                else
                    LOG_F(ERROR, "Could not reload work load file");
            }
            catch (const nlohmann::detail::parse_error&) {
                LOG_F(ERROR, "Could not reload work load file");
            }
        }

        if (auto startNode =
              frameGraph.acquireNode(frameGraph.getStageStartNode(FrameGraph::SIMULATION_STAGE),
                                     FrameGraph::SIMULATION_STAGE, simulationFrame);
            startNode) {
            garbageSystem.startGarbage(simulationFrame);
        }
        else
            break;
        runtimeStats.incrementFrame();
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

drv::Swapchain::OldSwapchinData Engine::recreateSwapchain() {
    std::unique_lock<std::mutex> lock(swapchainMutex);
    // images need to keep the same memory address
    return swapchain.recreate(physicalDevice, window);
}

void Engine::present(drv::SwapchainPtr swapchainPtr, FrameId frameId, uint32_t imageIndex,
                     uint32_t semaphoreIndex) {
    if (drv::is_null_ptr(swapchainPtr) || frameGraph.isStopped())
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
        if (auto startNode =
              frameGraph.acquireNode(frameGraph.getStageStartNode(FrameGraph::BEFORE_DRAW_STAGE),
                                     FrameGraph::BEFORE_DRAW_STAGE, beforeDrawFrame);
            startNode) {
            if (swapchainRecreationRequired) {
                std::unique_lock<std::mutex> lock(swapchainRecreationMutex);

                if (beforeDrawFrame > 0
                    && !frameGraph.waitForNode(frameGraph.getStageEndNode(FrameGraph::RECORD_STAGE),
                                               FrameGraph::RECORD_STAGE, beforeDrawFrame - 1)) {
                    assert(frameGraph.isStopped());
                    break;
                }

                releaseSwapchainResources();

                swapchainRecreationPossible = true;
                beforeDrawSwapchainCv.wait(lock);

                createSwapchainResources(swapchain);
                swapchainRecreationRequired = false;
            }
        }
        else
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
        AcquiredImageData swapchainData = mainRecord(recordFrame);
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
        if (auto endNode =
              frameGraph.acquireNode(frameGraph.getStageEndNode(FrameGraph::RECORD_STAGE),
                                     FrameGraph::RECORD_STAGE, recordFrame);
            endNode) {
            frameGraph.getGlobalExecutionQueue()->push(
              ExecutionPackage(ExecutionPackage::MessagePackage{
                ExecutionPackage::Message::FRAME_SUBMITTED, recordFrame, 0, nullptr}));
        }
        else {
            assert(frameGraph.isStopped());
            break;
        }
        recordFrame++;
    }
    // No node can be waiting for enqueue at this point (or they will never be enqueued)
    frameGraph.getGlobalExecutionQueue()->push(ExecutionPackage(
      ExecutionPackage::MessagePackage{ExecutionPackage::Message::QUIT, 0, 0, nullptr}));
}

class AccessValidationCallback final : public drv::ResourceStateTransitionCallback
{
 public:
    AccessValidationCallback(Engine* _engine, drv::QueuePtr _currentQueue,
                             FrameGraph::NodeId _nodeId, FrameId _frame)
      : engine(_engine),
        currentQueue(_currentQueue),
        nodeId(_nodeId),
        frame(_frame),
        semaphores(engine->garbageSystem.getAllocator<SemaphoreInfo>()),
        queues(engine->garbageSystem.getAllocator<QueueInfo>()) {}

    void requireSync(drv::QueuePtr srcQueue, drv::CmdBufferId srcSubmission, uint64_t srcFrameId,
                     drv::ResourceStateTransitionCallback::ConflictMode mode,
                     drv::PipelineStages::FlagType ongoingUsages) {
        drv::drv_assert(srcFrameId <= frame);
        if (srcQueue == currentQueue)
            return;
        FrameGraph::NodeId srcNode = engine->frameGraph.getNodeFromCmdBuffer(srcSubmission);
        switch (mode) {
            case drv::ResourceStateTransitionCallback::ConflictMode::NONE:
                return;
            case drv::ResourceStateTransitionCallback::ConflictMode::MUTEX: {
                LOG_F(
                  WARNING,
                  "Two asynchronous submissions try to use an exclusive resource (nodes %s -> %s). Either make the resource shared, or synchronize them.",
                  engine->frameGraph.getNode(srcNode)->getName().c_str(),
                  engine->frameGraph.getNode(nodeId)->getName().c_str());
                {
                    RuntimeStatsWriter writer = RUNTIME_STATS_WRITER;
                    writer->lastExecution
                      .invalidSharedResourceUsage[engine->frameGraph.getNode(srcNode)->getName()]
                      .push_back(engine->frameGraph.getNode(nodeId)->getName());
                }
                BREAK_POINT;
                break;
            }
            case drv::ResourceStateTransitionCallback::ConflictMode::ORDERED_ACCESS:
                drv::drv_assert(engine->frameGraph.hasEnqueueDependency(
                                  srcNode, nodeId, static_cast<uint32_t>(frame - srcFrameId)),
                                "Conflicting submissions have no enqueue dependency");
                {
                    StatsCacheWriter writer(engine->frameGraph.getStatsCacheHandle(srcSubmission));
                    writer->semaphore.append(ongoingUsages);
                }
        }
    }

    void registerSemaphore(drv::QueuePtr srcQueue, drv::CmdBufferId cmdBufferId,
                           drv::TimelineSemaphoreHandle semaphore, uint64_t srcFrameId,
                           uint64_t waitValue, drv::PipelineStages::FlagType semaphoreWaitStages,
                           drv::PipelineStages::FlagType waitMask, ConflictMode mode) override {
        requireSync(srcQueue, cmdBufferId, srcFrameId, mode, waitMask);
        drv::drv_assert(srcFrameId <= frame);
        if (srcQueue == currentQueue)
            return;
        bool found = false;
        for (uint32_t i = 0; i < semaphores.size() && !found; ++i) {
            if (semaphores[i].semaphore.ptr == semaphore.ptr) {
                semaphores[i].waitValue = std::max(semaphores[i].waitValue, waitValue);
                semaphores[i].semaphoreWaitStages |= semaphoreWaitStages;
                drv::drv_assert(semaphores[i].queue == srcQueue);
                found = true;
            }
        }
        if (!found)
            semaphores.push_back({semaphore, srcQueue, waitValue, semaphoreWaitStages});
    }

    void requireAutoSync(drv::QueuePtr srcQueue, drv::CmdBufferId cmdBufferId, uint64_t srcFrameId,
                         drv::PipelineStages::FlagType semaphoreWaitStages,
                         drv::PipelineStages::FlagType waitMask, ConflictMode mode,
                         AutoSyncReason reason) override {
        requireSync(srcQueue, cmdBufferId, srcFrameId, mode, waitMask);
        drv::drv_assert(srcFrameId <= frame);
        if (srcQueue == currentQueue)
            return;
        switch (reason) {
            case NO_SEMAPHORE:
                engine->runtimeStats.incrementNumGpuAutoSyncNoSemaphore();
                break;
            case INSUFFICIENT_SEMAPHORE:
                engine->runtimeStats.incrementNumGpuAutoSyncInsufficientSemaphore();
                break;
        }
        bool found = false;
        for (uint32_t i = 0; i < queues.size() && !found; ++i) {
            if (queues[i].queue == srcQueue) {
                queues[i].waitValue =
                  std::max(queues[i].waitValue, FrameGraph::get_semaphore_value(srcFrameId));
                queues[i].semaphoreWaitStages |= semaphoreWaitStages;
                found = true;
            }
        }
        if (!found) {
            FrameGraph::QueueSyncData data = engine->frameGraph.sync_queue(srcQueue, srcFrameId);
            queues.push_back(
              {srcQueue, std::move(data.semaphore), data.waitValue, semaphoreWaitStages});
        }
    }

    void filterSemaphores() {
        // erase semaphores, which from queues, which have auto sync
        semaphores.erase(std::remove_if(semaphores.begin(), semaphores.end(),
                                        [this](const SemaphoreInfo& info) {
                                            return std::find_if(queues.begin(), queues.end(),
                                                                [&](const QueueInfo& queueInfo) {
                                                                    return info.queue
                                                                           == queueInfo.queue;
                                                                })
                                                   != queues.end();
                                        }),
                         semaphores.end());
    }

    uint32_t getNumSemaphores() const { return uint32_t(semaphores.size() + queues.size()); }
    drv::TimelineSemaphorePtr getSemaphore(uint32_t index) const {
        if (index < semaphores.size())
            return semaphores[index].semaphore.ptr;
        return queues[index - semaphores.size()].autoSemaphore.ptr;
    }
    uint64_t getWaitValue(uint32_t index) const {
        if (index < semaphores.size())
            return semaphores[index].waitValue;
        return queues[index - semaphores.size()].waitValue;
    }
    drv::PipelineStages::FlagType getWaitStages(uint32_t index) const {
        if (index < semaphores.size())
            return semaphores[index].semaphoreWaitStages;
        return queues[index - semaphores.size()].semaphoreWaitStages;
    }

 private:
    struct SemaphoreInfo
    {
        drv::TimelineSemaphoreHandle semaphore;
        drv::QueuePtr queue;
        uint64_t waitValue;
        drv::PipelineStages::FlagType semaphoreWaitStages;
    };
    struct QueueInfo
    {
        drv::QueuePtr queue;
        drv::TimelineSemaphoreHandle autoSemaphore;
        uint64_t waitValue;
        drv::PipelineStages::FlagType semaphoreWaitStages;
    };

    Engine* engine;
    drv::QueuePtr currentQueue;
    FrameGraph::NodeId nodeId;
    FrameId frame;
    GarbageVector<SemaphoreInfo> semaphores;
    GarbageVector<QueueInfo> queues;
};

bool Engine::execute(ExecutionPackage&& package) {
    if (std::holds_alternative<ExecutionPackage::MessagePackage>(package.package)) {
        ExecutionPackage::MessagePackage& message =
          std::get<ExecutionPackage::MessagePackage>(package.package);
        switch (message.msg) {
            case ExecutionPackage::Message::FRAMEGRAPH_NODE_START_MARKER: {
                FrameGraph::NodeId nodeId = static_cast<FrameGraph::NodeId>(message.value1);
                FrameId frame = static_cast<FrameId>(message.value2);
                frameGraph.getNode(nodeId)->registerExecutionStart(frame);
            } break;
            case ExecutionPackage::Message::FRAMEGRAPH_NODE_FINISH_MARKER: {
                FrameGraph::NodeId nodeId = static_cast<FrameGraph::NodeId>(message.value1);
                FrameId frame = static_cast<FrameId>(message.value2);
                frameGraph.getNode(nodeId)->registerExecutionFinish(frame);
                frameGraph.executionFinished(nodeId, frame);
            } break;
            case ExecutionPackage::Message::FRAME_SUBMITTED: {
                FrameId frame = static_cast<FrameId>(message.value1);
                frameGraph.submitSignalFrameEnd(frame);
                break;
            }
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
        runtimeStats.incrementSubmissionCount();
        drv::CommandBufferPtr commandBuffers[2];
        uint32_t numCommandBuffers = 0;
        ExecutionPackage::CommandBufferPackage& cmdBuffer =
          std::get<ExecutionPackage::CommandBufferPackage>(package.package);

        AccessValidationCallback cb(
          this, cmdBuffer.queue,
          frameGraph.getNodeFromCmdBuffer(cmdBuffer.cmdBufferData.cmdBufferId), cmdBuffer.frameId);

        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> signalTimelineSemaphores(
          cmdBuffer.signalTimelineSemaphores.size() + 1, TEMPMEM);
        StackMemory::MemoryHandle<uint64_t> signalTimelineSemaphoreValues(
          cmdBuffer.signalTimelineSemaphores.size() + 1, TEMPMEM);
        StackMemory::MemoryHandle<drv::SemaphorePtr> waitSemaphores(cmdBuffer.waitSemaphores.size(),
                                                                    TEMPMEM);
        StackMemory::MemoryHandle<drv::PipelineStages::FlagType> waitSemaphoresStages(
          cmdBuffer.waitSemaphores.size(), TEMPMEM);
        for (uint32_t i = 0; i < cmdBuffer.signalTimelineSemaphores.size(); ++i) {
            signalTimelineSemaphores[i] = cmdBuffer.signalTimelineSemaphores[i].semaphore;
            signalTimelineSemaphoreValues[i] = cmdBuffer.signalTimelineSemaphores[i].signalValue;
            runtimeStats.incrementNumTimelineSemaphores();
        }
        uint32_t signalTimelineSemaphoreCount = uint32_t(cmdBuffer.signalTimelineSemaphores.size());
        if (cmdBuffer.signalManagedSemaphore) {
            cmdBuffer.signalManagedSemaphore.signal(cmdBuffer.signaledManagedSemaphoreValue);
            signalTimelineSemaphores[signalTimelineSemaphoreCount] =
              cmdBuffer.signalManagedSemaphore;
            signalTimelineSemaphoreValues[signalTimelineSemaphoreCount] =
              cmdBuffer.signaledManagedSemaphoreValue;
            signalTimelineSemaphoreCount++;
            runtimeStats.incrementNumTimelineSemaphores();
        }
        for (uint32_t i = 0; i < cmdBuffer.waitSemaphores.size(); ++i) {
            waitSemaphores[i] = cmdBuffer.waitSemaphores[i].semaphore;
            waitSemaphoresStages[i] =
              drv::get_image_usage_stages(cmdBuffer.waitSemaphores[i].imageUsages).stageFlags;
        }
        drv::ExecutionInfo executionInfo;
        executionInfo.numWaitSemaphores =
          static_cast<unsigned int>(cmdBuffer.waitSemaphores.size());
        executionInfo.waitSemaphores = waitSemaphores;
        executionInfo.waitStages = waitSemaphoresStages;
        drv::ResourceLocker::Lock resourceLock = {};
        if (!cmdBuffer.cmdBufferData.resourceUsages.empty()) {
            auto lock = resourceLocker.tryLock(&cmdBuffer.cmdBufferData.resourceUsages);
            if (lock.get() == drv::ResourceLocker::TryLockResult::SUCCESS)
                resourceLock = std::move(lock).getLock();
            else {
                // #ifdef DEBUG
                //                 LOG_F(WARNING, "Execution queue waits on CPU resource usage: %s",
                //                       cmdBuffer.cmdBufferData.getName());
                // #endif
                resourceLock =
                  resourceLocker.lock(&cmdBuffer.cmdBufferData.resourceUsages).getLock();
            }
        }
        {
            drv::StateCorrectionData correctionData;
            if (!drv::validate_and_apply_state_transitions(
                  getDevice(), cmdBuffer.queue, cmdBuffer.frameId,
                  cmdBuffer.cmdBufferData.cmdBufferId, cmdBuffer.signalManagedSemaphore,
                  cmdBuffer.signaledManagedSemaphoreValue,
                  cmdBuffer.cmdBufferData.semaphoreSrcStages, correctionData,
                  uint32_t(cmdBuffer.cmdBufferData.imageStates.size()),
                  cmdBuffer.cmdBufferData.imageStates.data(),
                  cmdBuffer.cmdBufferData.stateValidation ? cmdBuffer.cmdBufferData.statsCacheHandle
                                                          : nullptr,
                  &cb)) {
                if (cmdBuffer.cmdBufferData.stateValidation) {
                    // TODO turn breakpoint and log back on (in debug builds)
                    // LOG_F(ERROR, "Some resources are not in the expected state");
                    runtimeStats.corrigateSubmission(cmdBuffer.cmdBufferData.getName());
                    // BREAK_POINT;
                }
                else {
                    runtimeStats.incrementAllowedSubmissionCorrections();
                }
                OneTimeCmdBuffer<const drv::StateCorrectionData*> correctionCmdBuffer(
                  CMD_BUFFER_ID(), "correctionCmdBuffer", getSemaphorePool(), physicalDevice,
                  device, cmdBuffer.queue, getCommandBufferBank(), getGarbageSystem(),
                  [](const drv::StateCorrectionData* const& data,
                     drv::DrvCmdBufferRecorder* recorder) { recorder->corrigate(*data); },
                  frameGraph.get_semaphore_value(cmdBuffer.frameId));
                commandBuffers[numCommandBuffers++] =
                  correctionCmdBuffer.use(&correctionData).cmdBufferPtr;
            }
            if (!drv::is_null_ptr(cmdBuffer.cmdBufferData.cmdBufferPtr)) {
                commandBuffers[numCommandBuffers++] = cmdBuffer.cmdBufferData.cmdBufferPtr;
            }
        }
        cb.filterSemaphores();
        StackMemory::MemoryHandle<drv::TimelineSemaphorePtr> waitTimelineSemaphores(
          cmdBuffer.waitTimelineSemaphores.size() + cb.getNumSemaphores(), TEMPMEM);
        StackMemory::MemoryHandle<drv::PipelineStages::FlagType> waitTimelineSemaphoresStages(
          cmdBuffer.waitTimelineSemaphores.size() + cb.getNumSemaphores(), TEMPMEM);
        StackMemory::MemoryHandle<uint64_t> waitTimelineSemaphoresValues(
          cmdBuffer.waitTimelineSemaphores.size() + cb.getNumSemaphores(), TEMPMEM);
        for (uint32_t i = 0; i < cmdBuffer.waitTimelineSemaphores.size(); ++i) {
            waitTimelineSemaphores[i] = cmdBuffer.waitTimelineSemaphores[i].semaphore;
            waitTimelineSemaphoresValues[i] = cmdBuffer.waitTimelineSemaphores[i].waitValue;
            waitTimelineSemaphoresStages[i] =
              drv::get_image_usage_stages(cmdBuffer.waitTimelineSemaphores[i].imageUsages)
                .stageFlags;
        }
        for (uint32_t i = 0; i < cb.getNumSemaphores(); ++i) {
            uint32_t ind = i + uint32_t(cmdBuffer.waitTimelineSemaphores.size());
            waitTimelineSemaphores[ind] = cb.getSemaphore(i);
            waitTimelineSemaphoresStages[ind] = cb.getWaitStages(i);
            waitTimelineSemaphoresValues[ind] = cb.getWaitValue(i);
        }
        executionInfo.numCommandBuffers = numCommandBuffers;
        executionInfo.commandBuffers = commandBuffers;
        executionInfo.numSignalSemaphores =
          static_cast<unsigned int>(cmdBuffer.signalSemaphores.size());
        executionInfo.signalSemaphores = cmdBuffer.signalSemaphores.data();
        executionInfo.numWaitTimelineSemaphores = static_cast<unsigned int>(
          cmdBuffer.waitTimelineSemaphores.size() + cb.getNumSemaphores());
        executionInfo.waitTimelineSemaphores = waitTimelineSemaphores;
        executionInfo.timelineWaitValues = waitTimelineSemaphoresValues;
        executionInfo.timelineWaitStages = waitTimelineSemaphoresStages;
        executionInfo.numSignalTimelineSemaphores = signalTimelineSemaphoreCount;
        executionInfo.signalTimelineSemaphores = signalTimelineSemaphores;
        executionInfo.timelineSignalValues = signalTimelineSemaphoreValues;
        drv::execute(getDevice(), cmdBuffer.queue, 1, &executionInfo);
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

void Engine::readbackLoop(volatile bool* finished) {
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
    *finished = true;
    {
        // wait for execution queue to finish
        std::unique_lock<std::mutex> executionLock(executionMutex);
        drv::device_wait_idle(device);
        garbageSystem.releaseAll();
    }
}

void Engine::gameLoop() {
    RUNTIME_STAT_SCOPE(gameLoop);

    createSwapchainResources(swapchain);

    volatile bool readbackFinished = false;

    std::thread simulationThread(&Engine::simulationLoop, this);
    std::thread beforeDrawThread(&Engine::beforeDrawLoop, this);
    std::thread recordThread(&Engine::recordCommandsLoop, this);
    std::thread executeThread(&Engine::executeCommandsLoop, this);
    std::thread readbackThread(&Engine::readbackLoop, this, &readbackFinished);

    try {
        runtimeStats.initExecution();

        set_thread_name(&simulationThread, "simulation");
        set_thread_name(&beforeDrawThread, "beforeDraw");
        set_thread_name(&recordThread, "record");
        set_thread_name(&executeThread, "execute");
        set_thread_name(&readbackThread, "readback");

        entityManager.startFrameGraph(this);

        IWindow* w = window;
        while (!w->shouldClose()) {
            mainLoopKernel();
        }
        frameGraph.stopExecution(false);
        while (!readbackFinished) {
            mainLoopKernel();
        }
        simulationThread.join();
        beforeDrawThread.join();
        recordThread.join();
        executeThread.join();
        readbackThread.join();

        entityManager.exportToFile(fs::path{"prev_scene.json"});

        runtimeStats.stopExecution();
        runtimeStats.exportReport(launchArgs.reportFile);
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
    runtimeStats.incrementInputSample();
    static_cast<IWindow*>(window)->pollEvents();

    if (garbageSystem.getStartedFrame() != INVALID_FRAME) {
        if (swapchainRecreationPossible) {
            // must block here before destroying old swapchain to prevent another recreation before destroying the prev one
            std::unique_lock<std::mutex> swapchainLock(swapchainRecreationMutex);
            swapchainRecreationPossible = false;

            drv::Swapchain::OldSwapchinData oldSwapchain = recreateSwapchain();
            swapchainState = SwapchainState::UNKNOWN;

            garbageSystem.useGarbage(
              [&](Garbage* trashBin) { trashBin->release(std::move(oldSwapchain)); });

            beforeDrawSwapchainCv.notify_one();
        }
        else if (!swapchainRecreationRequired) {
            drv::Extent2D windowExtent = window->getResolution();
            if (windowExtent.width != 0 && windowExtent.height != 0) {
                if (windowExtent != swapchain.getCurrentEXtent()
                    || (swapchainState != SwapchainState::OK
                        && swapchainState != SwapchainState::UNKNOWN)) {
                    swapchainRecreationRequired = true;
                }
            }
        }
    }
}

void Engine::transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager,
                                FrameGraph::QueueId queue, FrameId frame,
                                FrameGraph::NodeHandle& nodeHandle,
                                ImageStager::StagerId stagerId) {
    struct Data
    {
        ImageStager* stager;
        ImageStager::StagerId stagerId;
        bool operator==(const Data& rhs) const {
            return stager == rhs.stager && stagerId == rhs.stagerId;
        }
        bool operator!=(const Data& rhs) const { return !(*this == rhs); }
        static void record(const Data& data, drv::DrvCmdBufferRecorder* recorder) {
            recorder->cmdImageBarrier({data.stager->getImage(),
                                       drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            data.stager->transferFromStager(recorder, data.stagerId);
        }
    } data = {&stager, stagerId};
    OneTimeCmdBuffer<Data> cmdBuffer(cmdBufferId, "engine_stager", getSemaphorePool(),
                                     getPhysicalDevice(), getDevice(), frameGraph.getQueue(queue),
                                     getCommandBufferBank(), getGarbageSystem(), Data::record,
                                     getFrameGraph().get_semaphore_value(frame));
    ExecutionPackage::CommandBufferPackage submission =
      make_submission_package(frameGraph.getQueue(queue), frame, cmdBuffer.use(std::move(data)),
                              getGarbageSystem(), ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager,
                                FrameGraph::QueueId queue, FrameId frame,
                                FrameGraph::NodeHandle& nodeHandle, ImageStager::StagerId stagerId,
                                uint32_t layer, uint32_t mip) {
    struct Data
    {
        ImageStager* stager;
        ImageStager::StagerId stagerId;
        uint32_t layer;
        uint32_t mip;
        bool operator==(const Data& rhs) const {
            return stager == rhs.stager && stagerId == rhs.stagerId && layer == rhs.layer
                   && mip == rhs.mip;
        }
        bool operator!=(const Data& rhs) const { return !(*this == rhs); }
        static void record(const Data& data, drv::DrvCmdBufferRecorder* recorder) {
            recorder->cmdImageBarrier({data.stager->getImage(),
                                       drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            data.stager->transferFromStager(recorder, data.stagerId, data.layer, data.mip);
        }
    } data = {&stager, stagerId, layer, mip};
    OneTimeCmdBuffer<Data> cmdBuffer(cmdBufferId, "engine_stager", getSemaphorePool(),
                                     getPhysicalDevice(), getDevice(), frameGraph.getQueue(queue),
                                     getCommandBufferBank(), getGarbageSystem(), Data::record,
                                     getFrameGraph().get_semaphore_value(frame));
    ExecutionPackage::CommandBufferPackage submission =
      make_submission_package(frameGraph.getQueue(queue), frame, cmdBuffer.use(std::move(data)),
                              getGarbageSystem(), ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager,
                                FrameGraph::QueueId queue, FrameId frame,
                                FrameGraph::NodeHandle& nodeHandle, ImageStager::StagerId stagerId,
                                const drv::ImageSubresourceRange& subres) {
    struct Data
    {
        ImageStager* stager;
        ImageStager::StagerId stagerId;
        const drv::ImageSubresourceRange* subres;
        bool operator==(const Data& rhs) const {
            return stager == rhs.stager && stagerId == rhs.stagerId && subres == rhs.subres;
        }
        bool operator!=(const Data& rhs) const { return !(*this == rhs); }
        static void record(const Data& data, drv::DrvCmdBufferRecorder* recorder) {
            recorder->cmdImageBarrier({data.stager->getImage(),
                                       drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            data.stager->transferFromStager(recorder, data.stagerId, *data.subres);
        }
    } data = {&stager, stagerId, &subres};
    OneTimeCmdBuffer<Data> cmdBuffer(cmdBufferId, "engine_stager", getSemaphorePool(),
                                     getPhysicalDevice(), getDevice(), frameGraph.getQueue(queue),
                                     getCommandBufferBank(), getGarbageSystem(), Data::record,
                                     getFrameGraph().get_semaphore_value(frame));
    ExecutionPackage::CommandBufferPackage submission =
      make_submission_package(frameGraph.getQueue(queue), frame, cmdBuffer.use(std::move(data)),
                              getGarbageSystem(), ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager,
                              FrameGraph::QueueId queue, FrameId frame,
                              FrameGraph::NodeHandle& nodeHandle, ImageStager::StagerId stagerId) {
    struct Data
    {
        ImageStager* stager;
        ImageStager::StagerId stagerId;
        bool operator==(const Data& rhs) const {
            return stager == rhs.stager && stagerId == rhs.stagerId;
        }
        bool operator!=(const Data& rhs) const { return !(*this == rhs); }
        static void record(const Data& data, drv::DrvCmdBufferRecorder* recorder) {
            recorder->cmdImageBarrier({data.stager->getImage(), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            data.stager->transferToStager(recorder, data.stagerId);
        }
    } data = {&stager, stagerId};
    OneTimeCmdBuffer<Data> cmdBuffer(cmdBufferId, "engine_stager", getSemaphorePool(),
                                     getPhysicalDevice(), getDevice(), frameGraph.getQueue(queue),
                                     getCommandBufferBank(), getGarbageSystem(), Data::record,
                                     getFrameGraph().get_semaphore_value(frame));
    ExecutionPackage::CommandBufferPackage submission =
      make_submission_package(frameGraph.getQueue(queue), frame, cmdBuffer.use(std::move(data)),
                              getGarbageSystem(), ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager,
                              FrameGraph::QueueId queue, FrameId frame,
                              FrameGraph::NodeHandle& nodeHandle, ImageStager::StagerId stagerId,
                              uint32_t layer, uint32_t mip) {
    struct Data
    {
        ImageStager* stager;
        ImageStager::StagerId stagerId;
        uint32_t layer;
        uint32_t mip;
        bool operator==(const Data& rhs) const {
            return stager == rhs.stager && stagerId == rhs.stagerId && layer == rhs.layer
                   && mip == rhs.mip;
        }
        bool operator!=(const Data& rhs) const { return !(*this == rhs); }
        static void record(const Data& data, drv::DrvCmdBufferRecorder* recorder) {
            recorder->cmdImageBarrier({data.stager->getImage(), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            data.stager->transferToStager(recorder, data.stagerId, data.layer, data.mip);
        }
    } data = {&stager, stagerId, layer, mip};
    OneTimeCmdBuffer<Data> cmdBuffer(cmdBufferId, "engine_stager", getSemaphorePool(),
                                     getPhysicalDevice(), getDevice(), frameGraph.getQueue(queue),
                                     getCommandBufferBank(), getGarbageSystem(), Data::record,
                                     getFrameGraph().get_semaphore_value(frame));
    ExecutionPackage::CommandBufferPackage submission =
      make_submission_package(frameGraph.getQueue(queue), frame, cmdBuffer.use(std::move(data)),
                              getGarbageSystem(), ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager,
                              FrameGraph::QueueId queue, FrameId frame,
                              FrameGraph::NodeHandle& nodeHandle, ImageStager::StagerId stagerId,
                              const drv::ImageSubresourceRange& subres) {
    struct Data
    {
        ImageStager* stager;
        ImageStager::StagerId stagerId;
        const drv::ImageSubresourceRange* subres;
        bool operator==(const Data& rhs) const {
            return stager == rhs.stager && stagerId == rhs.stagerId && subres == rhs.subres;
        }
        bool operator!=(const Data& rhs) const { return !(*this == rhs); }
        static void record(const Data& data, drv::DrvCmdBufferRecorder* recorder) {
            recorder->cmdImageBarrier({data.stager->getImage(), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            data.stager->transferToStager(recorder, data.stagerId, *data.subres);
        }
    } data = {&stager, stagerId, &subres};
    OneTimeCmdBuffer<Data> cmdBuffer(cmdBufferId, "engine_stager", getSemaphorePool(),
                                     getPhysicalDevice(), getDevice(), frameGraph.getQueue(queue),
                                     getCommandBufferBank(), getGarbageSystem(), Data::record,
                                     getFrameGraph().get_semaphore_value(frame));
    ExecutionPackage::CommandBufferPackage submission =
      make_submission_package(frameGraph.getQueue(queue), frame, cmdBuffer.use(std::move(data)),
                              getGarbageSystem(), ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}

void Engine::drawEntities(drv::DrvCmdBufferRecorder* recorder, drv::ImagePtr targetImage) {
    drv::TextureInfo targetInfo = drv::get_texture_info(targetImage);
    if (targetInfo.extent.width == 0 || targetInfo.extent.height == 0)
        return;
    for (const auto& entity : entitiesToDraw) {
        drv::TextureInfo textureInfo =
          drv::get_texture_info(entityManager.getTexture(entity.textureId));

        glm::vec2 clampedMin = glm::clamp(entity.relBottomLeft, glm::vec2(-1, -1), glm::vec2(1, 1));
        glm::vec2 clampedMax = glm::clamp(entity.relTopRight, glm::vec2(-1, -1), glm::vec2(1, 1));

        glm::vec2 size = entity.relTopRight - entity.relBottomLeft;

        glm::vec2 relMin = clampedMin - entity.relBottomLeft;
        relMin.x /= size.x;
        relMin.y /= size.y;
        glm::vec2 relMax = clampedMax - entity.relBottomLeft;
        relMax.x /= size.x;
        relMax.y /= size.y;

        drv::ImageBlit region;
        region.srcSubresource.aspectMask = drv::COLOR_BIT;
        region.srcSubresource.baseArrayLayer = 0;
        region.srcSubresource.layerCount = 1;
        region.srcSubresource.mipLevel = 0;
        region.dstSubresource.aspectMask = drv::COLOR_BIT;
        region.dstSubresource.baseArrayLayer = 0;
        region.dstSubresource.layerCount = 1;
        region.dstSubresource.mipLevel = 0;
        region.srcOffsets[0] = drv::Offset3D{int(floorf(relMin.x * textureInfo.extent.width)),
                                             int(floorf(relMin.y * textureInfo.extent.height)), 0};
        region.srcOffsets[1] = drv::Offset3D{int(ceil(relMax.x * textureInfo.extent.width)),
                                             int(ceil(relMax.y * textureInfo.extent.height)), 1};
        region.dstOffsets[0] =
          drv::Offset3D{int(floorf(((clampedMin.x * 0.5f) + 0.5f) * targetInfo.extent.width)),
                        int(floorf(((clampedMin.y * 0.5f) + 0.5f) * targetInfo.extent.height)), 0};
        region.dstOffsets[1] =
          drv::Offset3D{int(ceil(((clampedMax.x * 0.5f) + 0.5f) * targetInfo.extent.width)),
                        int(ceil(((clampedMax.y * 0.5f) + 0.5f) * targetInfo.extent.height)), 1};
        for (uint32_t i : {0u, 1u}) {
            if (region.dstOffsets[i].x < 0)
                region.dstOffsets[i].x = 0;
            if (region.dstOffsets[i].y < 0)
                region.dstOffsets[i].y = 0;
            if (region.dstOffsets[i].x > int(targetInfo.extent.width))
                region.dstOffsets[i].x = int(targetInfo.extent.width);
            if (region.dstOffsets[i].y > int(targetInfo.extent.height))
                region.dstOffsets[i].y = int(targetInfo.extent.height);
            if (region.srcOffsets[i].x < 0)
                region.srcOffsets[i].x = 0;
            if (region.srcOffsets[i].y < 0)
                region.srcOffsets[i].y = 0;
            if (region.srcOffsets[i].x > int(textureInfo.extent.width))
                region.srcOffsets[i].x = int(textureInfo.extent.width);
            if (region.srcOffsets[i].y > int(textureInfo.extent.height))
                region.srcOffsets[i].y = int(textureInfo.extent.height);
        }

        if (region.srcOffsets[0].x < region.srcOffsets[1].x
            && region.srcOffsets[0].y < region.srcOffsets[1].y
            && region.dstOffsets[0].x < region.dstOffsets[1].x
            && region.dstOffsets[0].y < region.dstOffsets[1].y) {
            recorder->cmdImageBarrier({entityManager.getTexture(entity.textureId),
                                       drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            recorder->cmdImageBarrier({targetImage, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                                       drv::ImageMemoryBarrier::AUTO_TRANSITION});
            recorder->cmdBlitImage(entityManager.getTexture(entity.textureId), targetImage, 1,
                                   &region, drv::ImageFilter::LINEAR);
        }
    }
}

Engine::AcquiredImageData Engine::mainRecord(FrameId frameId) {
    TemporalResourceLockerDescriptor resourceDesc;
    lockResources(resourceDesc, frameId);
    if (FrameGraph::NodeHandle nodeHandle = getFrameGraph().acquireNode(
          mainRecordNode, FrameGraph::RECORD_STAGE, frameId, resourceDesc);
        nodeHandle) {
        struct RecordData
        {
            Engine* engine;
            AcquiredImageData* swapChainData;
            FrameId frameId;
            static void record(const RecordData& data, drv::DrvCmdBufferRecorder* recorder) {
                for (const auto& entity : data.engine->entitiesToDraw)
                    data.engine->entityManager.prepareTexture(entity.textureId, recorder);
                data.engine->record(*data.swapChainData, recorder, data.frameId);
            }
            bool operator==(const RecordData& other) {
                return engine == other.engine && swapChainData == other.swapChainData;
            }
            bool operator!=(const RecordData& other) { return !(*this == other); }
        };
        std::sort(
          entitiesToDraw.begin(), entitiesToDraw.end(),
          [](const EntityRenderData& lhs, const EntityRenderData& rhs) { return lhs.z > rhs.z; });
        AcquiredImageData swapChainData = acquiredSwapchainImage(nodeHandle);
        if (!swapChainData)
            return swapChainData;
        RecordData recordData{this, &swapChainData, frameId};
        {
            OneTimeCmdBuffer<RecordData> cmdBuffer(
              CMD_BUFFER_ID(), "main_draw", getSemaphorePool(), getPhysicalDevice(), getDevice(),
              queueInfos.renderQueue.handle, getCommandBufferBank(), getGarbageSystem(),
              RecordData::record, getFrameGraph().get_semaphore_value(frameId));
            ExecutionPackage::CommandBufferPackage submission = make_submission_package(
              queueInfos.renderQueue.handle, frameId, cmdBuffer.use(std::move(recordData)),
              getGarbageSystem(), ResourceStateValidationMode::NEVER_VALIDATE);
            submission.waitSemaphores.push_back(
              {swapChainData.imageAvailableSemaphore,
               drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE | drv::IMAGE_USAGE_TRANSFER_DESTINATION});
            submission.signalSemaphores.push_back(swapChainData.renderFinishedSemaphore);
            nodeHandle.submit(queueInfos.renderQueue.id, std::move(submission));
        }

        entitiesToDraw.clear();
        return swapChainData;
    }
    return {};
}
