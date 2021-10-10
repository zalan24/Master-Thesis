#include "engine.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include <imgui.h>

#include <features.h>

#include <corecontext.h>
#include <util.hpp>

#include <drv.h>
#include <drverror.h>
#include <drvwindow.h>

#include <namethreads.h>
#include <perf_metrics.h>

#include <shadertypes.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image_write.h>

#include "execution_queue.h"
#include "imagestager.h"

bool EngineInputListener::processMouseButton(const Input::MouseButtenEvent& event) {
    if (event.buttonId == 0) {
        if (event.type == event.PRESS)
            clicking = true;
        else
            clicking = false;
    }
    return false;
}

bool EngineInputListener::processMouseMove(const Input::MouseMoveEvent& event) {
    mX = event.relX;
    mY = event.relY;
    return false;
}

bool EngineInputListener::processKeyboard(const Input::KeyboardEvent& event) {
    if (event.key == KEY_F10) {
        if (event.type == event.PRESS)
            perfCapture = true;
        return true;
    }
    else if (event.key == KEY_F11) {
        if (event.type == event.PRESS)
            toggleFreeCame = true;
        return true;
    }
    else if (event.key == KEY_F7) {
        if (event.type == event.PRESS)
            toggleRecording = true;
        return true;
    }
    else if (event.key == KEY_F) {
        if (event.type == event.PRESS)
            physicsFrozen = !physicsFrozen;
        return true;
    }
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
            // TODO reword command buffer usage;
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

CameraControlInfo FreeCamInput::popCameraControls() {
    FrameGraph::Clock::time_point currentTime = FrameGraph::Clock::now();
    float durationS =
      float(std::chrono::duration_cast<std::chrono::microseconds>(currentTime - lastSample).count())
      / 1000000.0f;
    lastSample = currentTime;
    if (hasPrevData) {
        CameraControlInfo ret;
        glm::vec3 targetSpeed;
        targetSpeed.x = (right ? 1 : 0) - (left ? 1 : 0);
        targetSpeed.y = up ? 1 : 0;
        targetSpeed.z = (forward ? 1 : 0) - (backward ? 1 : 0);

        ret.translation = speed * durationS;

        if (glm::dot(targetSpeed, targetSpeed) > 0)
            targetSpeed = glm::normalize(targetSpeed) * (boost ? fastSpeed : normalSpeed);

        float v = glm::length(speed - targetSpeed);

        if (v > 0) {
            if (drag.y != 0) {
                float c = v + drag.x / drag.y;
                v = c * exp(-drag.y * durationS) - drag.x / drag.y;
            }
            else
                v -= drag.x * durationS;

            if (v <= 0)
                speed = targetSpeed;
            else
                speed = glm::normalize(speed - targetSpeed) * v + targetSpeed;
        }

        ret.rotation = cameraRotate;
        cameraRotate = glm::vec2(0, 0);
        return ret;
    }
    else {
        cameraRotate = glm::vec2(0, 0);
        hasPrevData = 1;
        return {};
    }
}

bool FreeCamInput::processKeyboard(const Input::KeyboardEvent& event) {
    if (event.key == KEY_LEFT_SHIFT)
        boost = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_W)
        forward = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_S)
        backward = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_A)
        left = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_D)
        right = event.type != Input::KeyboardEvent::RELEASE;
    else if (event.key == KEY_E)
        up = event.type != Input::KeyboardEvent::RELEASE;
    return true;
}

bool FreeCamInput::processMouseMove(const Input::MouseMoveEvent& event) {
    cameraRotate += glm::vec2(event.dX, event.dY) * rotationSpeed;
    return true;
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
    ret.sharingType = drv::SharingType::EXCLUSIVE;
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
    timestampPool(device),
    timestampCmdBuffers(physicalDevice, device, config.maxFramesInFlight + 1, 64),
    swapchain(physicalDevice, device, window,
              get_swapchain_create_info(config, presentQueue.queue, renderQueue.queue)),
    eventPool(device),
    syncBlock(device, safe_cast<uint32_t>(config.maxFramesInFlight)),  // TODO why just 2?
    // maxFramesInFlight + 1 for readback stage
    frameGraph(physicalDevice, device, &garbageSystem, &resourceLocker, &eventPool, &semaphorePool,
               &timestampPool, trackingConfig, config.maxFramesInExecutionQueue,
               config.maxFramesInFlight + 1, config.slopHistorySize),
    runtimeStats(!launchArgs.clearRuntimeStats, launchArgs.runtimeStatsPersistanceBin,
                 launchArgs.runtimeStatsGameExportsBin, launchArgs.runtimeStatsCacheBin),
    physics(),
    entityManager(physicalDevice, device, &frameGraph, &physics, resourceFolders.textures),
    timestsampRingBuffer(config.maxFramesInFlight + 1) {
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

    queueInfos.computeQueue = {computeQueue.queue,
                               frameGraph.registerQueue(computeQueue.queue, "compute")};
    queueInfos.DtoHQueue = {DtoHQueue.queue, frameGraph.registerQueue(DtoHQueue.queue, "DtoH")};
    queueInfos.HtoDQueue = {HtoDQueue.queue, frameGraph.registerQueue(HtoDQueue.queue, "HtoD")};
    queueInfos.inputQueue = {inputQueue.queue, frameGraph.registerQueue(inputQueue.queue, "input")};
    queueInfos.presentQueue = {presentQueue.queue,
                               frameGraph.registerQueue(presentQueue.queue, "present")};
    queueInfos.renderQueue = {renderQueue.queue,
                              frameGraph.registerQueue(renderQueue.queue, "render")};

    inputSampleNode =
      frameGraph.addNode(FrameGraph::Node("sample_input", FrameGraph::SIMULATION_STAGE));
    physicsSimulationNode =
      frameGraph.addNode(FrameGraph::Node("physics_simulation", FrameGraph::SIMULATION_STAGE));
    presentFrameNode = frameGraph.addNode(
      FrameGraph::Node("presentFrame", FrameGraph::RECORD_STAGE | FrameGraph::EXECUTION_STAGE));
    mainRecordNode = frameGraph.addNode(
      FrameGraph::Node("mainRecord", FrameGraph::RECORD_STAGE | FrameGraph::EXECUTION_STAGE));
    acquireSwapchainNode = frameGraph.addNode(
      FrameGraph::Node("acquireSwapchain", FrameGraph::RECORD_STAGE | FrameGraph::EXECUTION_STAGE));

    drv::drv_assert(config.maxFramesInExecutionQueue >= 1,
                    "maxFramesInExecutionQueue must be at least 1");
    drv::drv_assert(config.maxFramesInFlight >= 2, "maxFramesInFlight must be at least 2");
    drv::drv_assert(config.maxFramesInFlight >= config.maxFramesInExecutionQueue,
                    "maxFramesInFlight must be at least the value of maxFramesInExecutionQueue");
    const uint32_t presentDepOffset = static_cast<uint32_t>(config.maxFramesInFlight - 1);
    garbageSystem.resize(frameGraph.getMaxFramesInFlight());

    frameGraph.addAllGpuCompleteDependency(presentFrameNode, FrameGraph::RECORD_STAGE,
                                           presentDepOffset);
    imGuiInputListener = window->createImGuiInputListener();
    inputManager.registerListener(imGuiInputListener.get(), 100);
    inputManager.registerListener(&mouseListener, 99);

    freecamListener = std::make_unique<FreeCamInput>();

    drv::sync_gpu_clock(drvInstance, physicalDevice, device);
    nextTimelineCalibration =
      drv::Clock::now() + std::chrono::milliseconds(firstTimelineCalibrationTimeMs);

    fs::path optionsPath = fs::path{launchArgs.engineOptionsFile};
    if (fs::exists(optionsPath)) {
        try {
            engineOptions.importFromFile(optionsPath);
        }
        catch (...) {
            engineOptions = {};
        }
    }
    // TODO this should be synced with vblanks
    frameEndFixPoint = FrameGraph::Clock::now();
    startupTime = FrameGraph::Clock::now();
    lastLatencyFlashClick = FrameGraph::Clock::now();
    frameHistory.resize(frameGraph.getMaxFramesInFlight());
    perFrameTempInfo.resize(frameGraph.getMaxFramesInFlight());
}

void Engine::buildFrameGraph() {
    frameGraph.addDependency(
      mainRecordNode, FrameGraph::CpuDependency{acquireSwapchainNode, FrameGraph::RECORD_STAGE,
                                                FrameGraph::RECORD_STAGE, 0});
    frameGraph.addDependency(presentFrameNode, FrameGraph::EnqueueDependency{mainRecordNode, 0});

    drv::drv_assert(renderEntitySystem.flag != 0, "Render entity system was not registered ");
    drv::drv_assert(physicsEntitySystem.flag != 0, "Physics entity system was not registered ");
    drv::drv_assert(emitterEntitySystem.flag != 0, "Emitter entity system was not registered ");
    drv::drv_assert(cameraEntitySystem.flag != 0, "Camera entity system was not registered ");
    drv::drv_assert(benchmarkEntitySystem.flag != 0, "Banchmark entity system was not registered ");

    entityManager.addEntityTemplate("dynobj",
                                    {physicsEntitySystem.flag | renderEntitySystem.flag, 0});
    entityManager.addEntityTemplate("camera",
                                    {physicsEntitySystem.flag | cameraEntitySystem.flag, 0});
    entityManager.addEntityTemplate("emitter",
                                    {emitterEntitySystem.flag | physicsEntitySystem.flag, 0});
    entityManager.addEntityTemplate("benchmark", {benchmarkEntitySystem.flag, 0});

    frameGraph.addDependency(
      physicsEntitySystem.nodeId,
      FrameGraph::CpuDependency{physicsSimulationNode, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::SIMULATION_STAGE, 0});
    frameGraph.addDependency(
      emitterEntitySystem.nodeId,
      FrameGraph::CpuDependency{physicsSimulationNode, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::SIMULATION_STAGE, 0});
    frameGraph.addDependency(
      renderEntitySystem.nodeId,
      FrameGraph::CpuDependency{cameraEntitySystem.nodeId, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::BEFORE_DRAW_STAGE, 0});
    frameGraph.addDependency(
      renderEntitySystem.nodeId,
      FrameGraph::CpuDependency{physicsSimulationNode, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::BEFORE_DRAW_STAGE, 0});
    frameGraph.addDependency(
      cameraEntitySystem.nodeId,
      FrameGraph::CpuDependency{benchmarkEntitySystem.nodeId, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::SIMULATION_STAGE, 0});
    frameGraph.addDependency(
      cameraEntitySystem.nodeId,
      FrameGraph::CpuDependency{inputSampleNode, FrameGraph::SIMULATION_STAGE,
                                FrameGraph::SIMULATION_STAGE, 0});
    frameGraph.addDependency(
      cameraEntitySystem.nodeId,
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

    // present node doesn't have measurable executions, so main render needs to be used instead
    frameGraph.build(inputSampleNode, FrameGraph::SIMULATION_STAGE, mainRecordNode);

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

void Engine::initImGui(drv::RenderPass* imGuiRenderpass) {
    imGuiIniter = std::make_unique<ImGuiIniter>(
      static_cast<IWindow*>(window), drvInstance, physicalDevice, device,
      queueInfos.renderQueue.handle, queueInfos.HtoDQueue.handle, imGuiRenderpass,
      frameGraph.getMaxFramesInFlight(), frameGraph.getMaxFramesInFlight());
}

void Engine::esBenchmark(EntityManager*, Engine* engine, FrameGraph::NodeHandle* nodeHandle,
                         FrameGraph::Stage, const EntityManager::EntitySystemParams& params,
                         Entity* entity, Entity::EntityId, FlexibleArray<Entity, 4>&) {
    auto prepareTimeItr = entity->extra.find("prepareTime");
    drv::drv_assert(prepareTimeItr != entity->extra.end(),
                    "Benchmark entity needs prepareTime component");
    auto periodItr = entity->extra.find("period");
    drv::drv_assert(periodItr != entity->extra.end(), "Benchmark entity needs period component");
    auto exitAfterItr = entity->extra.find("exitAfter");
    auto motionItr = entity->extraStr.find("motion");
    drv::drv_assert(motionItr != entity->extraStr.end(),
                    "Benchmark entity needs motion string component");
    float& timer = entity->extra["time"];
    timer += params.dt;
    const std::string ending = ".bin";
    bool isFromFile = motionItr->second.length() > ending.size()
                      && motionItr->second.compare(motionItr->second.length() - ending.length(),
                                                   ending.length(), ending)
                           == 0;
    if (isFromFile && motionItr->second != engine->loadedCameraMotion) {
        try {
            engine->benchmarkCameraMotion.importFromFile(fs::path{motionItr->second});
            drv::drv_assert(!engine->benchmarkCameraMotion.entries.empty(),
                            "Camera motions is empty");
        }
        catch (const std::exception& e) {
            LOG_F(ERROR, "Could not load camera motion: %s, reason: %s", motionItr->second.c_str(),
                  e.what());
            throw;
        }
        engine->loadedCameraMotion = motionItr->second;
    }
    float periodTime = periodItr->second;
    if (periodTime < 0) {
        drv::drv_assert(isFromFile, "Auto period only works with motions loaded from file");
        periodTime = engine->benchmarkCameraMotion.entries.back().timeMs / 1000.f;
    }
    float p = (timer - prepareTimeItr->second) / periodTime;
    engine->perFrameTempInfo[nodeHandle->getFrameId() % engine->perFrameTempInfo.size()]
      .benchmarkPeriod = p;
    if (exitAfterItr != entity->extra.end()) {
        if (p > exitAfterItr->second)
            engine->wantToQuit = true;
    }
    if (p >= 0) {
        p -= float(int(p));
        if (motionItr->second == "rotate") {
            entity->rotation = glm::quat(glm::vec3(0, p * float(M_PI) * 2.f, 0));
            entity->position = glm::vec3(0, 3, 0) - entity->rotation * glm::vec3(0, 0, 10.f);
        }
        else if (motionItr->second == "none") {
            // nothing to do
        }
        else if (motionItr->second == "intensifyingRotation") {
            auto minRotFreqItr = entity->extra.find("minRotFreq");
            drv::drv_assert(minRotFreqItr != entity->extra.end(),
                            "This benchmark entity needs minRotFreq component");
            auto maxRotFreqItr = entity->extra.find("maxRotFreq");
            drv::drv_assert(maxRotFreqItr != entity->extra.end(),
                            "This benchmark entity needs maxRotFreq component");
            auto rotAngleItr = entity->extra.find("rotAngle");
            drv::drv_assert(rotAngleItr != entity->extra.end(),
                            "This benchmark entity needs rotAngle component");
            auto rotMaxPowItr = entity->extra.find("rotMaxPow");
            drv::drv_assert(rotMaxPowItr != entity->extra.end(),
                            "This benchmark entity needs rotMaxPow component");
            float freq = lerp(minRotFreqItr->second, maxRotFreqItr->second, p * p * p);
            double s = sin(M_PI * 2 * double(freq * periodTime * p));
            double sgn = s > 0 ? 1 : -1;
            s = sgn
                * (1.0
                   - pow(1.0 - abs(s), lerp(1.0, double(rotMaxPowItr->second), double(p * p * p))));
            s = s * 0.5 + 0.5;
            float angle = float(lerp(0.0, double(rotAngleItr->second) / 180.0 * M_PI, s));
            entity->rotation = glm::quat(glm::vec3(0, angle, 0));
            // entity->position = glm::vec3(0, 3, 0) - entity->rotation * glm::vec3(0, 0, 10.f);
        }
        else {
            drv::drv_assert(isFromFile, "Unknown rotation");
            engine->benchmarkCameraMotion.interpolate(
              p * engine->benchmarkCameraMotion.entries.back().timeMs, entity->rotation,
              entity->position);
        }
    }
}

void Engine::esCamera(EntityManager* entityManger, Engine* engine, FrameGraph::NodeHandle* handle,
                      FrameGraph::Stage, const EntityManager::EntitySystemParams&, Entity* entity,
                      Entity::EntityId, FlexibleArray<Entity, 4>&) {
    if (!engine->isInFreecam()) {
        Entity::EntityId benchmarkId = entityManger->getByName("benchmark");
        if (benchmarkId != Entity::INVALID_ENTITY) {
            Entity* benchmark = entityManger->getById(benchmarkId);
            entity->position = benchmark->position;
            entity->rotation = benchmark->rotation;
        }
    }
    else {
        CameraControlInfo cameraControls =
          engine->perFrameTempInfo[handle->getFrameId() % engine->perFrameTempInfo.size()]
            .cameraControls;
        if (std::abs(cameraControls.rotation.x) > 0) {
            entity->rotation =
              glm::angleAxis(cameraControls.rotation.x, glm::vec3(0, 1, 0)) * entity->rotation;
        }
        if (std::abs(cameraControls.rotation.y) > 0) {
            const double upMinAngle = 0.1;
            double angle = double(cameraControls.rotation.y);
            glm::vec3 dir = entity->rotation * glm::vec3(0, 0, 1);
            double currentAngle = acos(double(dir.y));
            // double currentAngle = glm::pitch(entity->rotation) + M_PI / 2;
            if (angle < 0 && currentAngle + angle < upMinAngle)
                angle = upMinAngle - currentAngle;
            else if (angle > 0 && currentAngle + angle > M_PI - upMinAngle)
                angle = M_PI - upMinAngle - currentAngle;
            glm::quat rot = glm::angleAxis(static_cast<float>(angle),
                                           glm::normalize(glm::vec3(dir.z, 0, -dir.x)));
            entity->rotation = rot * entity->rotation;
        }

        entity->position += entity->rotation * cameraControls.translation;
    }

    auto brightnessItr = entity->extra.find("brightness");
    drv::drv_assert(brightnessItr != entity->extra.end(),
                    "Camera entity needs brightness component");
    auto sunDirXItr = entity->extra.find("sunDirX");
    drv::drv_assert(sunDirXItr != entity->extra.end(), "Camera entity needs sunDirX component");
    auto sunDirYItr = entity->extra.find("sunDirY");
    drv::drv_assert(sunDirYItr != entity->extra.end(), "Camera entity needs sunDirY component");
    auto sunDirZItr = entity->extra.find("sunDirZ");
    drv::drv_assert(sunDirZItr != entity->extra.end(), "Camera entity needs sunDirZ component");
    auto sunLightRItr = entity->extra.find("sunLightR");
    drv::drv_assert(sunLightRItr != entity->extra.end(), "Camera entity needs sunLightR component");
    auto sunLightGItr = entity->extra.find("sunLightG");
    drv::drv_assert(sunLightGItr != entity->extra.end(), "Camera entity needs sunLightG component");
    auto sunLightBItr = entity->extra.find("sunLightB");
    drv::drv_assert(sunLightBItr != entity->extra.end(), "Camera entity needs sunLightB component");
    auto ambientLightRItr = entity->extra.find("ambientLightR");
    drv::drv_assert(ambientLightRItr != entity->extra.end(),
                    "Camera entity needs ambientLightR component");
    auto ambientLightGItr = entity->extra.find("ambientLightG");
    drv::drv_assert(ambientLightGItr != entity->extra.end(),
                    "Camera entity needs ambientLightG component");
    auto ambientLightBItr = entity->extra.find("ambientLightB");
    drv::drv_assert(ambientLightBItr != entity->extra.end(),
                    "Camera entity needs ambientLightB component");

    drv::Extent2D extent = engine->window->getResolution();
    FrameGraph::Clock::time_point now = FrameGraph::Clock::now();
    RendererData& renderData =
      engine->perFrameTempInfo[handle->getFrameId() % engine->perFrameTempInfo.size()].renderData;
    renderData.latencyFlash =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - engine->lastLatencyFlashClick)
        .count()
      < 300;
    renderData.cameraRecord = engine->hasStartedRecording;
    float brightness = brightnessItr->second;
    renderData.sunDir =
      glm::normalize(vec3(sunDirXItr->second, sunDirYItr->second, sunDirZItr->second));
    renderData.sunLight =
      vec3(sunLightRItr->second, sunLightGItr->second, sunLightBItr->second) * brightness;
    renderData.ambientLight =
      vec3(ambientLightRItr->second, ambientLightGItr->second, ambientLightBItr->second)
      * brightness;
    renderData.eyePos = entity->position;
    renderData.eyeDir = static_cast<glm::mat3>(entity->rotation)[2];
    renderData.ratio = float(extent.width) / float(extent.height);
    renderData.cursorPos = engine->mouseListener.getMousePos() * 2.f - 1.f;

    if (engine->isRecording != engine->hasStartedRecording) {
        if (!engine->hasStartedRecording) {
            engine->cameraRecordStart = now;
            LOG_ENGINE("Camera recording has started");
        }
        else {
            fs::path recordsFolder{"cameraRecords"};
            if (!fs::exists(recordsFolder))
                fs::create_directories(recordsFolder);
            std::stringstream filename;
            auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            struct tm timeinfo;
            localtime_s(&timeinfo, &time);
            filename << "camera__" << std::put_time(&timeinfo, "%Y_%m_%d_%H_%M_%S") << ".bin";
            fs::path recordFile = recordsFolder / fs::path{filename.str()};
            fs::path recordFile2 = recordsFolder / fs::path{"latest.bin"};

            engine->cameraRecord.exportToFile(recordFile);
            engine->cameraRecord.exportToFile(recordFile2);
            LOG_ENGINE("Camera record saved to: %s", recordFile.string().c_str());
            engine->cameraRecord.entries.clear();
        }
        engine->hasStartedRecording = engine->isRecording;
    }
    if (engine->hasStartedRecording) {
        float timeMs = float(std::chrono::duration_cast<std::chrono::microseconds>(
                               now - engine->cameraRecordStart)
                               .count())
                       / 1000.f;
        engine->cameraRecord.entries.push_back(
          TransformRecordEntry{entity->rotation, entity->position, timeMs});
    }
}

void Engine::esEmitter(EntityManager*, Engine* engine, FrameGraph::NodeHandle*, FrameGraph::Stage,
                       const EntityManager::EntitySystemParams& params, Entity* entity,
                       Entity::EntityId, FlexibleArray<Entity, 4>& outEntities) {
    auto freqItr = entity->extra.find("freq");
    drv::drv_assert(freqItr != entity->extra.end(), "Emitter entity needs freq component");
    auto maxThetaItr = entity->extra.find("maxTheta");
    drv::drv_assert(maxThetaItr != entity->extra.end(), "Emitter entity needs maxTheta component");
    auto minSizeMulItr = entity->extra.find("minSizeMul");
    drv::drv_assert(minSizeMulItr != entity->extra.end(),
                    "Emitter entity needs minSizeMul component");
    auto maxSizeMulItr = entity->extra.find("maxSizeMul");
    drv::drv_assert(maxSizeMulItr != entity->extra.end(),
                    "Emitter entity needs maxSizeMul component");
    auto minSpeedItr = entity->extra.find("minSpeed");
    drv::drv_assert(minSpeedItr != entity->extra.end(), "Emitter entity needs minSpeed component");
    auto maxSpeedItr = entity->extra.find("maxSpeed");
    drv::drv_assert(maxSpeedItr != entity->extra.end(), "Emitter entity needs maxSpeed component");
    auto baseMassItr = entity->extra.find("baseMass");
    drv::drv_assert(baseMassItr != entity->extra.end(), "Emitter entity needs baseMass component");
    auto placeDistItr = entity->extra.find("placeDist");
    drv::drv_assert(placeDistItr != entity->extra.end(),
                    "Emitter entity needs placeDist component");
    float& emitTimer = entity->extra["emitTimer"];
    float& counter = entity->extra["emitCounter"];
    if (emitTimer < 0)
        emitTimer = 0;
    emitTimer += params.dt;
    float threshold = 1.0f / freqItr->second;
    // this should avoid placing entities in the same position in case, they are emitted in the same frame
    float timePassedSinceEmitted = 0;
    while (threshold <= emitTimer) {
        emitTimer -= threshold;

        if (!entity->hidden) {
            float phi = engine->genFloat01() * float(M_PI) * 2;
            float theta = engine->genFloat01() * maxThetaItr->second;
            glm::quat orientation =
              glm::quat(glm::vec3(0, 0, phi)) * glm::quat(glm::vec3(theta, 0, 0));
            orientation = entity->rotation * orientation;
            glm::vec3 dir = orientation * glm::vec3(0, 0, 1);
            float velocity = lerp(minSpeedItr->second, maxSpeedItr->second, engine->genFloat01());
            float scale = lerp(minSizeMulItr->second, maxSizeMulItr->second, engine->genFloat01());

            Entity ent;
            ent.name = entity->name + "_emitted_" + std::to_string(counter);
            ent.templateName = "dynobj";
            ent.parentName = "";
            ent.albedo = entity->albedo;
            ent.velocity = entity->velocity + dir * velocity;
            ent.position =
              entity->position + dir * placeDistItr->second + ent.velocity * timePassedSinceEmitted;
            ent.scale = entity->scale * scale;
            ent.rotation = orientation;
            ent.modelName = entity->modelName;
            ent.specular = entity->specular;
            ent.bumpyness = entity->bumpyness * ent.scale.x * ent.scale.y * ent.scale.z;
            ent.bumpScale = entity->bumpScale;
            ent.mandelbrot = entity->mandelbrot;
            if (baseMassItr->second > 0) {
                ent.mass = baseMassItr->second * ent.scale.x * ent.scale.y * ent.scale.z;
                drv::drv_assert(ent.mass > 0);
            }
            else
                ent.mass = baseMassItr->second;
            ent.hidden = false;

            outEntities.push_back(std::move(ent));
        }
        timePassedSinceEmitted += threshold;
        counter += 1;
    }
    //    Entity entity;
    //                            std::uniform_int_distribution<int> distribution(1,6);
    // int dice_roll = distribution(generator);
}

void Engine::esPhysics(EntityManager* entityManager, Engine* engine, FrameGraph::NodeHandle*,
                       FrameGraph::Stage, const EntityManager::EntitySystemParams&, Entity* entity,
                       Entity::EntityId id, FlexibleArray<Entity, 4>&) {
    if (!entity->rigidBody)
        return;
    Physics::RigidBodyState state =
      engine->physics.getRigidBodyState(static_cast<RigidBodyPtr>(entity->rigidBody));
    entity->position = state.position;
    entity->rotation = state.rotation;
    entity->velocity = state.velocity;

    if (entity->position.y < -10)
        entityManager->removeEntity(id);
}

void Engine::esBeforeDraw(EntityManager*, Engine* engine, FrameGraph::NodeHandle*,
                          FrameGraph::Stage, const EntityManager::EntitySystemParams&,
                          Entity* entity, Entity::EntityId, FlexibleArray<Entity, 4>&) {
    if (entity->hidden)
        return;

    EntityRenderData data;

    glm::vec3 position = entity->position;
    glm::vec3 scale = entity->scale;
    glm::quat rotation = entity->rotation;

    data.albedo = entity->albedo;
    data.specular = entity->specular;
    data.bumpyness = entity->bumpyness;
    data.bumpScale = entity->bumpScale;
    data.mandelbrot = entity->mandelbrot;
    data.shape = entity->modelName;
    glm::mat4 translationTm = glm::translate(glm::mat4(1.f), position);
    glm::mat4 scaleTm = glm::scale(glm::mat4(1.f), scale);
    glm::mat4 rotTm = static_cast<glm::mat4>(rotation);
    data.modelTm = translationTm * rotTm * scaleTm;

    engine->entitiesToDraw.push_back(std::move(data));
}

void Engine::initPhysicsEntitySystem() {
    physicsEntitySystem = entityManager.addEntitySystem(
      "entityPhysics", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esPhysics);
    emitterEntitySystem = entityManager.addEntitySystem(
      "emitterES", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esEmitter);
    benchmarkEntitySystem = entityManager.addEntitySystem(
      "benchmarkES", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esBenchmark);
}

void Engine::initRenderEntitySystem() {
    renderEntitySystem = entityManager.addEntitySystem(
      "entityRender", FrameGraph::BEFORE_DRAW_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, true}, esBeforeDraw);
}

void Engine::initCursorEntitySystem() {
    cameraEntitySystem = entityManager.addEntitySystem(
      "cameraES", FrameGraph::SIMULATION_STAGE,
      {EntityManager::EntitySystemInfo::ENGINE_SYSTEM, false}, esCamera);
}

Engine::~Engine() {
    captureImageStager.clear();
    captureImage.close();
    fs::path optionsPath = fs::path{launchArgs.engineOptionsFile};
    if (!fs::exists(optionsPath))
        fs::create_directories(optionsPath.parent_path());
    engineOptions.exportToFile(optionsPath);
    garbageSystem.releaseAll();
    inputManager.unregisterListener(&mouseListener);
    inputManager.unregisterListener(imGuiInputListener.get());
    if (inFreeCam)
        inputManager.unregisterListener(freecamListener.get());
    if (!benchmarkData.empty()) {
        fs::path benchmarkFolder{"benchmarks"};
        if (!fs::exists(benchmarkFolder))
            fs::create_directories(benchmarkFolder);
        std::stringstream filename;
        auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        struct tm timeinfo;
        localtime_s(&timeinfo, &time);
        filename << "benchmark__" << std::put_time(&timeinfo, "%Y_%m_%d_%H_%M_%S") << ".csv";
        fs::path benchmarkFile = benchmarkFolder / fs::path{"benchmark.csv"};
        fs::path benchmarkFileCopy = benchmarkFolder / fs::path{filename.str()};
        std::ofstream benchmarkOut(benchmarkFile.string().c_str());
        std::ofstream benchmarkOutCopy(benchmarkFileCopy.string().c_str());
        if (benchmarkOut || benchmarkOutCopy) {
            while (!benchmarkData.empty()) {
                BenchmarkData entry = benchmarkData.front();
                benchmarkData.pop_front();
                if (benchmarkOut)
                    benchmarkOut << std::setprecision(16) << entry.period << std::setprecision(3)
                                 << ", " << entry.fps << ", " << entry.latency << ", "
                                 << entry.latencySlop << ", " << entry.cpuWork << ", "
                                 << entry.execWork << ", " << entry.deviceWork << ", "
                                 << entry.workTime << ", " << entry.missRate << ", "
                                 << entry.execDelay << ", " << entry.deviceDelay << ", "
                                 << entry.potentialFps << ", " << entry.potentialCpuFps << ", "
                                 << entry.potentialExecFps << ", " << entry.potentialDeviceFps
                                 << std::endl;
                if (benchmarkOutCopy)
                    benchmarkOutCopy
                      << std::setprecision(16) << entry.period << std::setprecision(3) << ", "
                      << entry.fps << ", " << entry.latency << ", " << entry.latencySlop << ", "
                      << entry.cpuWork << ", " << entry.execWork << ", " << entry.deviceWork << ", "
                      << entry.workTime << ", " << entry.missRate << ", " << entry.execDelay << ", "
                      << entry.deviceDelay << ", " << entry.potentialFps << ", "
                      << entry.potentialCpuFps << ", " << entry.potentialExecFps << ", "
                      << entry.potentialDeviceFps << std::endl;
            }
        }
        else
            LOG_F(ERROR, "Could not export benchmark data");
    }
    LOG_ENGINE("Engine closed");
}

bool Engine::simulatePhysics(FrameId frameId) {
    FrameGraph::NodeHandle physicsHandle =
      frameGraph.acquireNode(physicsSimulationNode, FrameGraph::SIMULATION_STAGE, frameId);
    if (!physicsHandle)
        return false;
    const auto& frameInfo = perFrameTempInfo[frameId % perFrameTempInfo.size()];
    if (frameId > 0) {
        physics.stepSimulation(float(isFrozen() ? 0 : frameInfo.deltaTimeSec), 20, 0.0016f);
    }
    return true;
}

bool Engine::sampleInput(FrameId frameId) {
    FrameGraph::NodeHandle inputHandle =
      frameGraph.acquireNode(inputSampleNode, FrameGraph::SIMULATION_STAGE, frameId);
    if (!inputHandle)
        return false;

    float beforeSleep = genNormalDistribution(engineOptions.manualWorkload_beforeInputAvg,
                                              engineOptions.manualWorkload_beforeInputStdDiv);
    if (beforeSleep > 0)
        FrameGraph::busy_sleep(
          std::chrono::microseconds(static_cast<long long>(beforeSleep * 1000.f)));

    FrameGraphSlops::ExtendedLatencyInfo latencyInfo;
    {
        std::unique_lock<std::mutex> lock(latencyInfoMutex);
        latencyInfo = latestLatencyInfo;
    }
    double desiredSlop = double(engineOptions.desiredSlop);
    int64_t desiredSlopNs = int64_t(desiredSlop * 1000000.0);
    double intervalLenMs =
      1000.0
      / double(engineOptions.targetRefreshRate > 15 ? engineOptions.targetRefreshRate : 15.f);
    int64_t intervalLenNs = int64_t(intervalLenMs * 1000000.0);

    double refreshTimeMs = frameOffsetAvgMs;
    bool limitedFps = engineOptions.refreshMode == EngineOptions::LIMITED
                      || engineOptions.refreshMode == EngineOptions::DISCRETIZED;
    double headRoomMs = 0;
    if (limitedFps && refreshTimeMs < intervalLenMs) {
        headRoomMs = intervalLenMs - refreshTimeMs;
        refreshTimeMs = intervalLenMs;
    }
    FrameGraph::Clock::time_point estimatedPrevEnd = latencyInfo.finishTime;
    for (FrameId frame = latencyInfo.frame + 1; frame < frameId; ++frame) {
        const FrameHistoryInfo& info = frameHistory[frame % frameHistory.size()];
        estimatedPrevEnd += info.duration;
        if (estimatedPrevEnd > info.estimatedEnd)
            estimatedPrevEnd -= std::min(info.headRoom, estimatedPrevEnd - info.estimatedEnd);
    }
    FrameGraph::Clock::time_point estimatedCurrentEnd =
      estimatedPrevEnd + std::chrono::nanoseconds(int64_t(refreshTimeMs * 1000000.0));
    if (engineOptions.refreshMode == EngineOptions::DISCRETIZED) {
        int64_t sinceReferencePointNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          estimatedCurrentEnd - frameEndFixPoint)
                                          .count();
        int64_t mod = sinceReferencePointNs % intervalLenNs;
        // Can't save this frame, even by sacrificing the entire latency pool
        if (desiredSlopNs < mod || int64_t(headRoomMs * 1000000.0) < mod)
            sinceReferencePointNs += intervalLenNs;
        sinceReferencePointNs -= mod;
        estimatedCurrentEnd = frameEndFixPoint + std::chrono::nanoseconds(sinceReferencePointNs);
    }

    if (engineOptions.latencyReduction && fpsStats.hasInfo()
        && frameId > frameGraph.getMaxFramesInFlight()) {
        double sleepTimeMs = 0;

        if (!engineOptions.manualLatencyReduction) {
            double estimatedWork = worTimeAfterInputAvgMs;

            FrameGraph::Clock::time_point now = FrameGraph::Clock::now();
            // recovery from errors
            if (estimatedCurrentEnd > now + std::chrono::seconds(1))
                estimatedCurrentEnd = now + std::chrono::seconds(1);
            if (estimatedCurrentEnd < now)
                estimatedCurrentEnd =
                  now + std::chrono::nanoseconds(int64_t(estimatedWork * 1000000.0));

            double targetDuration =
              double(std::chrono::duration_cast<std::chrono::nanoseconds>(estimatedCurrentEnd - now)
                       .count())
              / 1000000.0;

            sleepTimeMs = targetDuration - estimatedWork;
            if (engineOptions.refreshMode == EngineOptions::LIMITED)
                sleepTimeMs -= std::max(0.0, desiredSlop - headRoomMs);
            else
                sleepTimeMs -= desiredSlop;
        }
        else {
            sleepTimeMs = double(engineOptions.manualSleepTime);
        }
        sleepTimeMs = std::max(std::min(sleepTimeMs, 100.0), 0.0);
        waitTimeStats.feed(sleepTimeMs);
        if (sleepTimeMs > 0) {
            auto timer = inputHandle.getLatencySleepTimer();
            FrameGraph::busy_sleep(std::chrono::microseconds(int64_t(sleepTimeMs * 1000.0)));
        }
    }
    else {
        waitTimeStats.feed(0);
    }
    std::chrono::nanoseconds duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(estimatedCurrentEnd - estimatedPrevEnd);
    frameHistory[frameId % frameHistory.size()] = {
      duration, estimatedCurrentEnd, std::chrono::nanoseconds(int64_t(headRoomMs * 1000000.0))};
    {
        // Try to acquire new input data
        std::unique_lock<std::mutex> lock(inputWaitMutex);
        mainKernelCv.notify_one();
        waitForInputCv.wait_for(lock, std::chrono::microseconds(500));
    }
    {
        std::unique_lock<std::mutex> lock(inputSamplingMutex);
        Input::InputEvent event;
        while (input.popEvent(event))
            inputManager.feedInput(std::move(event));
    }

    perFrameTempInfo[frameId % perFrameTempInfo.size()].cameraControls = {};
    if (mouseListener.popToggleFreeCame()) {
        if (!inFreeCam) {
            inputManager.registerListener(freecamListener.get(), 98);
            inFreeCam = true;
        }
        else {
            inputManager.unregisterListener(freecamListener.get());
            inFreeCam = false;
        }
    }
    if (mouseListener.popToggleRecording())
        isRecording = !isRecording;
    CameraControlInfo cameraControls = freecamListener->popCameraControls();
    if (inFreeCam)
        perFrameTempInfo[frameId % perFrameTempInfo.size()].cameraControls =
          std::move(cameraControls);

    return true;
}

void Engine::simulationLoop() {
    RUNTIME_STAT_SCOPE(simulationLoop);
    loguru::set_thread_name("simulate");
    LOG_ENGINE("Simulation thread started");
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

        frameGraph.initFrame(simulationFrame);

        if (auto startNode =
              frameGraph.acquireNode(frameGraph.getStageStartNode(FrameGraph::SIMULATION_STAGE),
                                     FrameGraph::SIMULATION_STAGE, simulationFrame);
            startNode) {
            garbageSystem.startGarbage(simulationFrame);
            const auto& prevInfo = perFrameTempInfo[(simulationFrame + perFrameTempInfo.size() - 1)
                                                    % perFrameTempInfo.size()];
            auto& info = perFrameTempInfo[simulationFrame % perFrameTempInfo.size()] = {};
            info.frameStartTime = FrameGraph::Clock::now();
            info.deltaTimeSec = double(std::chrono::duration_cast<std::chrono::microseconds>(
                                         info.frameStartTime - prevInfo.frameStartTime)
                                         .count())
                                / 1000000.0;
        }
        else
            break;
        runtimeStats.incrementFrame();
        if (!simulatePhysics(simulationFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        if (!sampleInput(simulationFrame)) {
            assert(frameGraph.isStopped());
            break;
        }
        entityManager.setFrozen(isFrozen());
        if (mouseListener.popNeedPerfCapture()) {
            createPerformanceCapture(simulationFrame);
        }
        if (mouseListener.isClicking())
            lastLatencyFlashClick = FrameGraph::Clock::now();
        simulate(simulationFrame);
        if (auto endNode =
              frameGraph.acquireNode(frameGraph.getStageEndNode(FrameGraph::SIMULATION_STAGE),
                                     FrameGraph::SIMULATION_STAGE, simulationFrame);
            endNode) {
            float afterSleep = genNormalDistribution(engineOptions.manualWorkload_afterInputAvg,
                                                     engineOptions.manualWorkload_afterInputStdDiv);
            if (afterSleep > 0)
                FrameGraph::busy_sleep(
                  std::chrono::microseconds(static_cast<long long>(afterSleep * 1000.f)));
        }
        else {
            assert(frameGraph.isStopped());
            break;
        }
        simulationFrame++;
    }
    LOG_ENGINE("Simulation thread stopped");
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
        drv::AcquireResult result;
        {
            auto timer = acquiringNodeHandle.getSlopTimer();
            result = swapchain.acquire(imageIndex,
                                       syncBlock.imageAvailableSemaphores[acquireImageSemaphoreId]);
        }
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
    LOG_ENGINE("Before draw thread started");
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
    LOG_ENGINE("Before draw thread stopped");
}

void Engine::recordCommandsLoop() {
    RUNTIME_STAT_SCOPE(recordLoop);
    loguru::set_thread_name("record");
    LOG_ENGINE("Record commands thread started");
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
                  ->push(ExecutionPackage(
                    recordFrame, presentHandle.getNodeId(),
                    ExecutionPackage::PresentPackage{recordFrame, swapchainData.imageIndex,
                                                     swapchainData.semaphoreIndex,
                                                     swapchainData.swapchain}));
            }
        }
        if (auto endNode =
              frameGraph.acquireNode(frameGraph.getStageEndNode(FrameGraph::RECORD_STAGE),
                                     FrameGraph::RECORD_STAGE, recordFrame);
            endNode) {
            frameGraph.getGlobalExecutionQueue()->push(ExecutionPackage(
              recordFrame, endNode.getNodeId(),
              ExecutionPackage::MessagePackage{ExecutionPackage::Message::FRAME_SUBMITTED,
                                               recordFrame, 0, nullptr}));
        }
        else {
            assert(frameGraph.isStopped());
            break;
        }
        recordFrame++;
    }
    // No node can be waiting for enqueue at this point (or they will never be enqueued)
    frameGraph.getGlobalExecutionQueue()->push(ExecutionPackage(
      recordFrame, INVALID_NODE,
      ExecutionPackage::MessagePackage{ExecutionPackage::Message::QUIT, 0, 0, nullptr}));
    LOG_ENGINE("Record commands thread stopped");
}

class AccessValidationCallback final : public drv::ResourceStateTransitionCallback
{
 public:
    AccessValidationCallback(Engine* _engine, drv::QueuePtr _currentQueue, NodeId _nodeId,
                             FrameId _frame)
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
        NodeId srcNode = engine->frameGraph.getNodeFromCmdBuffer(srcSubmission);
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
    NodeId nodeId;
    FrameId frame;
    GarbageVector<SemaphoreInfo> semaphores;
    GarbageVector<QueueInfo> queues;
};

bool Engine::execute(ExecutionPackage&& package) {
    bool recursiveExecution = false;
    drv::Clock::time_point startTime = drv::Clock::now();
    drv::CmdBufferId cmdBufferId = drv::CmdBufferId(-1);
    bool needEndMarker = false;
    if (std::holds_alternative<ExecutionPackage::MessagePackage>(package.package)) {
        ExecutionPackage::MessagePackage& message =
          std::get<ExecutionPackage::MessagePackage>(package.package);
        switch (message.msg) {
            case ExecutionPackage::Message::FRAMEGRAPH_NODE_START_MARKER: {
                NodeId nodeId = static_cast<NodeId>(message.value1);
                FrameId frame = static_cast<FrameId>(message.value2);
                frameGraph.getNode(nodeId)->registerExecutionStart(frame);
            } break;
            case ExecutionPackage::Message::FRAMEGRAPH_NODE_FINISH_MARKER: {
                NodeId nodeId = static_cast<NodeId>(message.value1);
                FrameId frame = static_cast<FrameId>(message.value2);
                frameGraph.getNode(nodeId)->registerExecutionFinish(frame);
                frameGraph.executionFinished(nodeId, frame);
                needEndMarker = true;
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
        if (nextTimelineCalibration < drv::Clock::now()) {
            drv::sync_gpu_clock(drvInstance, physicalDevice, device);
            nextTimelineCalibration =
              drv::Clock::now() + std::chrono::milliseconds(otherTimelineCalibrationTimeMs);
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
        runtimeStats.incrementSubmissionCount();
        drv::CommandBufferPtr commandBuffers[4];
        uint32_t numCommandBuffers = 0;
        ExecutionPackage::CommandBufferPackage& cmdBuffer =
          std::get<ExecutionPackage::CommandBufferPackage>(package.package);
        cmdBufferId = cmdBuffer.cmdBufferData.cmdBufferId;

        bool needTimestamps = drv::timestamps_supported(device, cmdBuffer.queue);
        SubmissionTimestampsInfo submissionTimestampInfo;

        if (needTimestamps) {
            TimestampCmdBufferPool::CmdBufferInfo info = timestampCmdBuffers.acquire(
              drv::get_queue_family(device, cmdBuffer.queue), cmdBuffer.frameId);
            commandBuffers[numCommandBuffers++] = info.cmdBuffer;
            submissionTimestampInfo.beginTimestampBufferIndex = info.index;
        }

        NodeId nodeId = frameGraph.getNodeFromCmdBuffer(cmdBuffer.cmdBufferData.cmdBufferId);

        AccessValidationCallback cb(this, cmdBuffer.queue, nodeId, cmdBuffer.frameId);

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
                  uint32_t(cmdBuffer.cmdBufferData.bufferStates.size()),
                  cmdBuffer.cmdBufferData.bufferStates.data(),
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
        if (needTimestamps) {
            TimestampCmdBufferPool::CmdBufferInfo info = timestampCmdBuffers.acquire(
              drv::get_queue_family(device, cmdBuffer.queue), cmdBuffer.frameId);
            commandBuffers[numCommandBuffers++] = info.cmdBuffer;
            submissionTimestampInfo.endTimestampBufferIndex = info.index;
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
        float execSleep = genNormalDistribution(engineOptions.manualWorkload_execInputAvg,
                                                engineOptions.manualWorkload_execInputStdDiv);
        if (execSleep > 0)
            FrameGraph::busy_sleep(
              std::chrono::microseconds(static_cast<long long>(execSleep * 1000.f)));
        drv::execute(getDevice(), cmdBuffer.queue, 1, &executionInfo);
        if (needTimestamps) {
            submissionTimestampInfo.submissionTime = drv::Clock::now();
            submissionTimestampInfo.node = nodeId;
            submissionTimestampInfo.queue = cmdBuffer.queue;
            submissionTimestampInfo.queueId = cmdBuffer.queueId;
            submissionTimestampInfo.submission = cmdBuffer.cmdBufferData.cmdBufferId;
            timestsampRingBuffer[cmdBuffer.frameId % timestsampRingBuffer.size()].push_back(
              std::move(submissionTimestampInfo));
        }
    }
    else if (std::holds_alternative<ExecutionPackage::RecursiveQueue>(package.package)) {
        ExecutionPackage::RecursiveQueue& queue =
          std::get<ExecutionPackage::RecursiveQueue>(package.package);
        ExecutionPackage p;
        recursiveExecution = true;
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
    if (!recursiveExecution && !needEndMarker)
        frameGraph.feedExecutionTiming(package.nodeId, package.frame, package.creationTime,
                                       startTime, drv::Clock::now(), cmdBufferId);
    return true;
}

void Engine::executeCommandsLoop() {
    RUNTIME_STAT_SCOPE(executionLoop);
    loguru::set_thread_name("execution");
    LOG_ENGINE("Execution thread started");
    std::unique_lock<std::mutex> executionLock(executionMutex);
    while (true) {
        ExecutionPackage package;
        ExecutionQueue* executionQueue = frameGraph.getGlobalExecutionQueue();
        executionQueue->waitForPackage();
        while (executionQueue->pop(package))
            if (!execute(std::move(package)))
                return;
    }
    LOG_ENGINE("Execution thread stopped");
}

void Engine::readbackLoop(volatile bool* finished) {
    RUNTIME_STAT_SCOPE(readbackLoop);
    loguru::set_thread_name("readback");
    LOG_ENGINE("Readback thread started");
    FrameId readbackFrame = 0;
    while (!frameGraph.isStopped()) {
        if (!frameGraph.startStage(FrameGraph::READBACK_STAGE, readbackFrame))
            break;
        readback(readbackFrame);

        bool performCapture =
          perfCaptureFrame != INVALID_FRAME
          && perfCaptureFrame + frameGraph.getMaxFramesInFlight() <= readbackFrame;

        TemporalResourceLockerDescriptor resourceDesc;
        ImageStager::StagerId stagerId = 0;
        if (performCapture) {
            stagerId = captureImageStager.getStagerId(perfCaptureFrame);
            captureImageStager.lockResource(resourceDesc, ImageStager::DOWNLOAD, stagerId);
        }

        if (auto handle =
              frameGraph.acquireNode(frameGraph.getStageEndNode(FrameGraph::READBACK_STAGE),
                                     FrameGraph::READBACK_STAGE, readbackFrame, resourceDesc);
            handle) {
            for (const auto& itr :
                 timestsampRingBuffer[readbackFrame % timestsampRingBuffer.size()]) {
                drv::PipelineStages trackedStages =
                  timestampCmdBuffers.getTrackedStages(drv::get_queue_family(device, itr.queue));
                uint32_t count = trackedStages.getStageCount();
                StackMemory::MemoryHandle<drv::Clock::time_point> startTimes(count, TEMPMEM);
                StackMemory::MemoryHandle<drv::Clock::time_point> endTimes(count, TEMPMEM);
                timestampCmdBuffers.readbackTimestamps(itr.queue, itr.beginTimestampBufferIndex,
                                                       startTimes);
                timestampCmdBuffers.readbackTimestamps(itr.queue, itr.endTimestampBufferIndex,
                                                       endTimes);
                FrameGraph::Node::DeviceTiming timing;
                timing.frameId = readbackFrame;
                timing.queue = itr.queue;
                timing.queueId = itr.queueId;
                timing.submissionId = itr.submission;
                timing.submitted = itr.submissionTime;
                timing.finish = endTimes[0];
                // LOG_ENGINE("Device timings at frame: %llu", readbackFrame);
                // for (uint32_t i = 0; i < count; ++i) {
                //     LOG_ENGINE(" - start: %lld end: %lld",
                //                std::chrono::duration_cast<std::chrono::milliseconds>(
                //                  startTimes[i] - itr.submissionTime)
                //                  .count(),
                //                std::chrono::duration_cast<std::chrono::milliseconds>(
                //                  endTimes[i] - itr.submissionTime)
                //                  .count());
                // }
                for (uint32_t i = 1; i < count; ++i)
                    if (timing.finish < endTimes[i])
                        timing.finish = endTimes[i];
                timing.start = itr.submissionTime;
                bool startInitialized = false;
                for (uint32_t i = 0; i < count; ++i) {
                    drv::PipelineStages::PipelineStageFlagBits stage = trackedStages.getStage(i);
                    if (stage == drv::PipelineStages::TOP_OF_PIPE_BIT)
                        continue;
                    if (!startInitialized || startTimes[i] < timing.start) {
                        timing.start = startTimes[i];
                        startInitialized = true;
                    }
                }
                drv::drv_assert(startInitialized, "Could not read start time");
                frameGraph.getNode(itr.node)->registerDeviceTiming(readbackFrame,
                                                                   std::move(timing));
            }
            timestsampRingBuffer[readbackFrame % timestsampRingBuffer.size()].clear();

            FrameGraphSlops::ExtendedLatencyInfo prevLatencyInfo = latestLatencyInfo;
            {
                std::unique_lock<std::mutex> lock(latencyInfoMutex);
                latestLatencyInfo = frameGraph.processSlops(readbackFrame);
            }
            if (latestLatencyInfo.frame != INVALID_FRAME)
                drv::drv_assert(latestLatencyInfo.info.inputNodeInfo.totalSlopNs
                                  <= latestLatencyInfo.info.inputNodeInfo.latencyNs,
                                "Slop cannot be greater than latency");

            // const FrameGraph::Node* simStart =
            //   frameGraph.getNode(frameGraph.getStageStartNode(FrameGraph::SIMULATION_STAGE));
            // const FrameGraph::Node* presentStart = frameGraph.getNode(presentFrameNode);

            double frameTimeMs = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          latestLatencyInfo.finishTime - prevLatencyInfo.finishTime)
                                          .count())
                                 / 1000000.0;

            fpsStats.feed(1000.0 / frameTimeMs);
            double frameOffsetMs = double(latestLatencyInfo.info.nextFrameOffsetNs) / 1000000.0;
            theoreticalFpsStats.feed(1000.0 / frameOffsetMs);
            double workAfterInputMs =
              double(latestLatencyInfo.info.inputNodeInfo.workTimeNs) / 1000000.0;
            double workMs = double(latestLatencyInfo.frameLatencyInfo.asyncWorkNs) / 1000000.0;
            double cpuWorkMs = double(latestLatencyInfo.info.cpuWorkNs) / 1000000.0;
            double cpuOffsetMs = double(latestLatencyInfo.info.cpuNextFrameOffsetNs) / 1000000.0;
            double execWorkMs = double(latestLatencyInfo.info.execWorkNs) / 1000000.0;
            double execOffsetMs = double(latestLatencyInfo.info.execNextFrameOffsetNs) / 1000000.0;
            double deviceWorkMs = double(latestLatencyInfo.info.deviceWorkNs) / 1000000.0;
            double deviceOffsetMs =
              double(latestLatencyInfo.info.deviceNextFrameOffsetNs) / 1000000.0;
            double latencyMs = double(latestLatencyInfo.info.inputNodeInfo.latencyNs) / 1000000.0;
            double slopMs = double(latestLatencyInfo.info.inputNodeInfo.totalSlopNs) / 1000000.0;
            // double cpuLimitMs = cpuWorkMs - cpuOverlapMs;
            // double execLimitMs = execWorkMs - execOverlapMs;
            // double deviceLimitMs = deviceWorkMs - deviceOverlapMs;
            cpuWorkStats.feed(cpuWorkMs);
            cpuOffsetStats.feed(cpuOffsetMs);
            execWorkStats.feed(execWorkMs);
            execOffsetStats.feed(execOffsetMs);
            deviceWorkStats.feed(deviceWorkMs);
            deviceOffsetStats.feed(deviceOffsetMs);
            latencyStats.feed(latencyMs);
            slopStats.feed(slopMs);
            perFrameSlopStats.feed(double(latestLatencyInfo.frameLatencyInfo.perFrameSlopNs)
                                   / 1000000.0);
            workStats.feed(workMs);
            double execDelayMs = double(latestLatencyInfo.frameLatencyInfo.execDelayNs) / 1000000.0;
            double deviceDelayMs =
              double(latestLatencyInfo.frameLatencyInfo.deviceDelayNs) / 1000000.0;
            execDelayStats.feed(execDelayMs);
            deviceDelayStats.feed(deviceDelayMs);
            if (latestLatencyInfo.frame < frameGraph.getMaxFramesInFlight()) {
                worTimeAfterInputAvgMs = workAfterInputMs;
                frameOffsetAvgMs = frameOffsetMs;
            }
            else {
                worTimeAfterInputAvgMs = lerp(workAfterInputMs, worTimeAfterInputAvgMs,
                                              double(engineOptions.workTimeSmoothing));
                frameOffsetAvgMs =
                  lerp(frameOffsetMs, frameOffsetAvgMs, double(engineOptions.workTimeSmoothing));
                // refreshTimeCpuAvgMs =
                //   lerp(cpuLimitMs, refreshTimeCpuAvgMs, double(engineOptions.workTimeSmoothing));
                // refreshTimeExecAvgMs =
                //   lerp(execLimitMs, refreshTimeExecAvgMs, double(engineOptions.workTimeSmoothing));
                // refreshTimeDeviceAvgMs = lerp(deviceLimitMs, refreshTimeDeviceAvgMs,
                //                               double(engineOptions.workTimeSmoothing));
            }
            // estimatedFrameEndTimes[latestLatencyInfo.frame % estimatedFrameEndTimes.size()] =
            //   latestLatencyInfo.finishTime;
            // FrameId prevIndex = (latestLatencyInfo.frame + estimatedFrameEndTimes.size() - 1)
            //                     % estimatedFrameEndTimes.size();
            // const auto expectedDuration =
            //   expectedFrameDurations[latestLatencyInfo.frame % expectedFrameDurations.size()];
            // const FrameGraph::Clock::time_point expectedFinishTime =
            //   estimatedFrameEndTimes[prevIndex] + expectedDuration;
            double skippedOrDelayed = 0;
            switch (engineOptions.refreshMode) {
                case EngineOptions::UNLIMITED:
                    // skippedOrDelayed =
                    //   (slopMs < double(engineOptions.desiredSlop) / 4.0) ? 1.0 : 0.0;
                    // break;
                case EngineOptions::LIMITED:
                    skippedOrDelayed = -1;
                    break;
                case EngineOptions::DISCRETIZED: {
                    double intervalLenMs = 1000.0 / double(engineOptions.targetRefreshRate);
                    int64_t intervalLenNs = int64_t(intervalLenMs * 1000000.0);
                    int64_t sinceReferencePointNs =
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                        latestLatencyInfo.finishTime - frameEndFixPoint)
                        .count();
                    int64_t completedInterval = sinceReferencePointNs / intervalLenNs;
                    skippedOrDelayed =
                      std::max(double(completedInterval - completedFrameIntervalId - 1), 0.0);
                    completedFrameIntervalId = completedInterval;
                    break;
                }
            }
            skippedDelayed.feed(skippedOrDelayed);

            BenchmarkData benchmarkEntry;
            benchmarkEntry.period =
              perFrameTempInfo[readbackFrame % perFrameTempInfo.size()].benchmarkPeriod;
            if (benchmarkEntry.period >= 0) {
                benchmarkEntry.fps = float(1000.0 / frameTimeMs);
                benchmarkEntry.latency = float(latencyMs);
                benchmarkEntry.latencySlop = float(slopMs);
                benchmarkEntry.cpuWork = float(cpuWorkMs);
                benchmarkEntry.execWork = float(execWorkMs);
                benchmarkEntry.deviceWork = float(deviceWorkMs);
                benchmarkEntry.workTime = float(workMs);
                benchmarkEntry.missRate = float(skippedDelayed.getAvg());
                benchmarkEntry.execDelay = float(execDelayMs);
                benchmarkEntry.deviceDelay = float(deviceDelayMs);
                benchmarkEntry.potentialFps = 1000.0f / float(frameOffsetMs);
                benchmarkEntry.potentialCpuFps = 1000.0f / float(cpuOffsetMs);
                benchmarkEntry.potentialExecFps = 1000.0f / float(execOffsetMs);
                benchmarkEntry.potentialDeviceFps = 1000.0f / float(deviceOffsetMs);
                benchmarkData.push_back(std::move(benchmarkEntry));
            }

            fs::path capturesFolder = fs::path{"captures"};
            if (perFrameTempInfo[readbackFrame % perFrameTempInfo.size()].captureHappening)
                captureLatencyInfo = latestLatencyInfo;

            if (performCapture) {
                PerformanceCaptureData capture =
                  generatePerfCapture(readbackFrame, captureLatencyInfo);
                if (!fs::exists(capturesFolder))
                    fs::create_directories(capturesFolder);
                try {
                    std::stringstream filename;

                    auto time =
                      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    struct tm timeinfo;
                    localtime_s(&timeinfo, &time);
                    filename << "capture__" << std::put_time(&timeinfo, "%Y_%m_%d_%H_%M_%S") << "_"
                             << perfCaptureFrame;

                    drv::DeviceSize size;
                    drv::DeviceSize rowPitch;
                    drv::DeviceSize arrayPitch;
                    drv::DeviceSize depthPitch;
                    captureImageStager.getMemoryData(stagerId, 0, 0, size, rowPitch, arrayPitch,
                                                     depthPitch);
                    std::vector<uint32_t> pixels(size / 4);
                    captureImageStager.getData(pixels.data(), 0, 0, stagerId, handle.getLock());
                    drv::TextureInfo texInfo =
                      drv::get_texture_info(captureImage.get().getImage(0));
                    std::vector<uint8_t> decodedPixels(texInfo.extent.width * texInfo.extent.height
                                                       * 3);
                    for (uint32_t y = 0; y < texInfo.extent.height; ++y) {
                        for (uint32_t x = 0; x < texInfo.extent.width; ++x) {
                            uint32_t p = pixels[y * rowPitch / 4 + x];
                            // uint8_t a = uint8_t(p >> 24);
                            uint8_t r = uint8_t(p >> 16);
                            uint8_t g = uint8_t(p >> 8);
                            uint8_t b = uint8_t(p);
                            decodedPixels[(y * texInfo.extent.width + x) * 3 + 0] = r;
                            decodedPixels[(y * texInfo.extent.width + x) * 3 + 1] = g;
                            decodedPixels[(y * texInfo.extent.width + x) * 3 + 2] = b;
                        }
                    }
                    std::string imageFileName = filename.str() + ".png";
                    std::string captureImageFile =
                      (capturesFolder / fs::path{imageFileName}).string();
                    stbi_write_png(captureImageFile.c_str(), int(texInfo.extent.width),
                                   int(texInfo.extent.height), 3, decodedPixels.data(),
                                   int(texInfo.extent.width * 3));

                    fs::path capturePath = capturesFolder / fs::path{filename.str() + ".html"};
                    generate_capture_file(capturesFolder / fs::path{"lastCapture.html"}, &capture,
                                          imageFileName);
                    generate_capture_file(capturePath, &capture, imageFileName);
                    LOG_ENGINE("Performance capture of frame %llu, saved to %s", perfCaptureFrame,
                               capturePath.string().c_str());
                }
                catch (const std::exception& e) {
                    LOG_F(ERROR, "Could not capture frame: %s", e.what());
                }
                perfCaptureFrame = INVALID_FRAME;
            }

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
        LOG_ENGINE("Doing some cleanup...");
        // wait for execution queue to finish
        std::unique_lock<std::mutex> executionLock(executionMutex);
        drv::device_wait_idle(device);
        garbageSystem.releaseAll();
    }
    LOG_ENGINE("Readback thread stopped");
}

void Engine::gameLoop() {
    RUNTIME_STAT_SCOPE(gameLoop);

    createSwapchainResources(swapchain);

    volatile bool readbackFinished = false;

    LOG_ENGINE("Starting loop threads...");
    std::thread simulationThread(&Engine::simulationLoop, this);
    std::thread beforeDrawThread(&Engine::beforeDrawLoop, this);
    std::thread recordThread(&Engine::recordCommandsLoop, this);
    std::thread executeThread(&Engine::executeCommandsLoop, this);
    std::thread readbackThread(&Engine::readbackLoop, this, &readbackFinished);

    try {
        LOG_ENGINE("Initing runtime stats...");
        runtimeStats.initExecution();

        LOG_ENGINE("Naming threads...");
        set_thread_name(&simulationThread, "simulation");
        set_thread_name(&beforeDrawThread, "beforeDraw");
        set_thread_name(&recordThread, "record");
        set_thread_name(&executeThread, "execute");
        set_thread_name(&readbackThread, "readback");

        LOG_ENGINE("Starting entity manager...");
        entityManager.startFrameGraph(this);

        IWindow* w = window;
        while (!w->shouldClose() && !wantToQuit) {
            mainLoopKernel();
        }
        LOG_ENGINE("Sending stop signal to frameGraph...");
        frameGraph.stopExecution(false);
        while (!readbackFinished) {
            mainLoopKernel();
        }
        LOG_ENGINE("Joining threads...");
        simulationThread.join();
        beforeDrawThread.join();
        recordThread.join();
        executeThread.join();
        readbackThread.join();

        drv::sync_gpu_clock(drvInstance, physicalDevice, device);

        LOG_ENGINE("Exporting stuff...");
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
    LOG_ENGINE("Game loop successfully ended");
}

void Engine::mainLoopKernel() {
    std::unique_lock<std::mutex> lock(mainKernelMutex);
    mainKernelCv.wait_for(lock, std::chrono::milliseconds(4));
    runtimeStats.incrementInputSample();
    {
        std::unique_lock<std::mutex> samplingLock(inputSamplingMutex);
        static_cast<IWindow*>(window)->pollEvents();
    }
    waitForInputCv.notify_one();

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

void Engine::transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                                FrameId frame, FrameGraph::NodeHandle& nodeHandle,
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
      make_submission_package(frameGraph.getQueue(queue), queue, frame, nodeHandle.getNodeId(),
                              cmdBuffer.use(std::move(data)), getGarbageSystem(),
                              ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                                FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                                ImageStager::StagerId stagerId, uint32_t layer, uint32_t mip) {
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
      make_submission_package(frameGraph.getQueue(queue), queue, frame, nodeHandle.getNodeId(),
                              cmdBuffer.use(std::move(data)), getGarbageSystem(),
                              ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferFromStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                                FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                                ImageStager::StagerId stagerId,
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
      make_submission_package(frameGraph.getQueue(queue), queue, frame, nodeHandle.getNodeId(),
                              cmdBuffer.use(std::move(data)), getGarbageSystem(),
                              ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                              FrameId frame, FrameGraph::NodeHandle& nodeHandle,
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
      make_submission_package(frameGraph.getQueue(queue), queue, frame, nodeHandle.getNodeId(),
                              cmdBuffer.use(std::move(data)), getGarbageSystem(),
                              ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                              FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                              ImageStager::StagerId stagerId, uint32_t layer, uint32_t mip) {
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
      make_submission_package(frameGraph.getQueue(queue), queue, frame, nodeHandle.getNodeId(),
                              cmdBuffer.use(std::move(data)), getGarbageSystem(),
                              ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}
void Engine::transferToStager(drv::CmdBufferId cmdBufferId, ImageStager& stager, QueueId queue,
                              FrameId frame, FrameGraph::NodeHandle& nodeHandle,
                              ImageStager::StagerId stagerId,
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
      make_submission_package(frameGraph.getQueue(queue), queue, frame, nodeHandle.getNodeId(),
                              cmdBuffer.use(std::move(data)), getGarbageSystem(),
                              ResourceStateValidationMode::NEVER_VALIDATE);
    nodeHandle.submit(queue, std::move(submission));
}

Engine::AcquiredImageData Engine::mainRecord(FrameId frameId) {
    TemporalResourceLockerDescriptor resourceDesc;
    lockResources(resourceDesc, frameId);
    AcquiredImageData swapChainData;

    if (FrameGraph::NodeHandle nodeHandle =
          getFrameGraph().acquireNode(acquireSwapchainNode, FrameGraph::RECORD_STAGE, frameId);
        nodeHandle) {
        swapChainData = acquiredSwapchainImage(nodeHandle);
        if (!swapChainData)
            return swapChainData;
        drv::Extent3D swapchainExtent = drv::get_texture_info(swapChainData.image).extent;
        if (!captureImage
            || drv::get_texture_info(captureImage.get().getImage(0)).extent != swapchainExtent) {
            drv::ImageSet::ImageInfo transferImageInfo;
            transferImageInfo.imageId = drv::ImageId("captureTransferTex");
            transferImageInfo.format = drv::get_texture_info(swapChainData.image).format;
            transferImageInfo.extent = swapchainExtent;
            transferImageInfo.mipLevels = 1;
            transferImageInfo.arrayLayers = 1;
            transferImageInfo.sampleCount = drv::SampleCount::SAMPLE_COUNT_1;
            transferImageInfo.usage =
              drv::ImageCreateInfo::TRANSFER_DST_BIT | drv::ImageCreateInfo::TRANSFER_SRC_BIT;
            transferImageInfo.type = drv::ImageCreateInfo::TYPE_2D;
            captureImageStager.clear();
            captureImage = createResource<drv::ImageSet>(
              getPhysicalDevice(), getDevice(),
              std::vector<drv::ImageSet::ImageInfo>{transferImageInfo},
              drv::ImageSet::PreferenceSelector(drv::MemoryType::DEVICE_LOCAL_BIT,
                                                drv::MemoryType::DEVICE_LOCAL_BIT));
            captureImageStager = ImageStager(this, captureImage.get().getImage(0),
                                             getMaxFramesInFlight(), ImageStager::DOWNLOAD);
        }
        perFrameTempInfo[frameId % perFrameTempInfo.size()].captureImage =
          captureImage.get().getImage(0);
    }
    else
        return {};

    if (FrameGraph::NodeHandle nodeHandle = getFrameGraph().acquireNode(
          mainRecordNode, FrameGraph::RECORD_STAGE, frameId, resourceDesc);
        nodeHandle) {
        struct RecordData
        {
            Engine* engine;
            AcquiredImageData* swapChainData;
            FrameId frameId;
            bool captureImage;
            drv::ImagePtr captureImageTarget;
            ImageStager* captureImageStager;
            static void record(const RecordData& data, drv::DrvCmdBufferRecorder* _recorder) {
                EngineCmdBufferRecorder recorder(_recorder);
                // for (const auto& entity : data.engine->entitiesToDraw)
                //     data.engine->entityManager.prepareTexture(entity.textureId, recorder);
                data.engine->record(*data.swapChainData, &recorder, data.frameId);
                if (data.captureImage) {
                    recorder.cmdImageBarrier({data.swapChainData->image,
                                              drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                              drv::ImageMemoryBarrier::AUTO_TRANSITION});
                    recorder.cmdImageBarrier({data.captureImageTarget,
                                              drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                                              drv::ImageMemoryBarrier::AUTO_TRANSITION});
                    drv::ImageCopyRegion region;
                    region.dstOffset = {0, 0, 0};
                    region.srcOffset = {0, 0, 0};
                    region.extent = drv::get_texture_info(data.captureImageTarget).extent;
                    region.srcSubresource.aspectMask = drv::COLOR_BIT;
                    region.srcSubresource.baseArrayLayer = 0;
                    region.srcSubresource.layerCount = 1;
                    region.srcSubresource.mipLevel = 0;
                    region.dstSubresource.aspectMask = drv::COLOR_BIT;
                    region.dstSubresource.baseArrayLayer = 0;
                    region.dstSubresource.layerCount = 1;
                    region.dstSubresource.mipLevel = 0;
                    recorder.cmdCopyImage(data.swapChainData->image, data.captureImageTarget, 1,
                                          &region);
                    recorder.cmdImageBarrier({data.captureImageTarget,
                                              drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                              drv::ImageMemoryBarrier::AUTO_TRANSITION});
                    data.captureImageStager->transferToStager(
                      recorder, data.captureImageStager->getStagerId(data.frameId));
                    recorder.cmdImageBarrier({data.swapChainData->image, drv::IMAGE_USAGE_PRESENT,
                                              drv::ImageMemoryBarrier::AUTO_TRANSITION});
                }
            }
            bool operator==(const RecordData& other) {
                return engine == other.engine && swapChainData == other.swapChainData;
            }
            bool operator!=(const RecordData& other) { return !(*this == other); }
        };
        // std::sort(
        //   entitiesToDraw.begin(), entitiesToDraw.end(),
        //   [](const EntityRenderData& lhs, const EntityRenderData& rhs) { return lhs.z > rhs.z; });
        bool needCaptureImage =
          perFrameTempInfo[frameId % perFrameTempInfo.size()].captureHappening;
        drv::ImagePtr captureImageTarget =
          perFrameTempInfo[frameId % perFrameTempInfo.size()].captureImage;
        RecordData recordData{
          this, &swapChainData, frameId, needCaptureImage, captureImageTarget, &captureImageStager};
        {
            OneTimeCmdBuffer<RecordData> cmdBuffer(
              CMD_BUFFER_ID(), "main_draw", getSemaphorePool(), getPhysicalDevice(), getDevice(),
              queueInfos.renderQueue.handle, getCommandBufferBank(), getGarbageSystem(),
              RecordData::record, getFrameGraph().get_semaphore_value(frameId));
            ExecutionPackage::CommandBufferPackage submission = make_submission_package(
              queueInfos.renderQueue.handle, queueInfos.renderQueue.id, frameId,
              nodeHandle.getNodeId(), cmdBuffer.use(std::move(recordData)), getGarbageSystem(),
              ResourceStateValidationMode::NEVER_VALIDATE);
            submission.waitSemaphores.push_back(
              {swapChainData.imageAvailableSemaphore,
               drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE | drv::IMAGE_USAGE_TRANSFER_DESTINATION,
               drv::IMAGE_USAGE_TRANSFER_SOURCE});
            submission.signalSemaphores.push_back(swapChainData.renderFinishedSemaphore);
            nodeHandle.submit(queueInfos.renderQueue.id, std::move(submission));
        }

        entitiesToDraw.clear();
    }
    else
        return {};
    return swapChainData;
}

Engine::ImGuiIniter::ImGuiIniter(IWindow* _window, drv::InstancePtr instance,
                                 drv::PhysicalDevicePtr _physicalDevice,
                                 drv::LogicalDevicePtr _device, drv::QueuePtr _renderQueue,
                                 drv::QueuePtr transferQueue, drv::RenderPass* renderpass,
                                 uint32_t minSwapchainImages, uint32_t swapchainImages)
  : window(_window) {
    window->initImGui(instance, _physicalDevice, _device, _renderQueue, transferQueue, renderpass,
                      minSwapchainImages, swapchainImages);
}

Engine::ImGuiIniter::~ImGuiIniter() {
    window->closeImGui();
}

void Engine::recordImGui(const AcquiredImageData&, EngineCmdBufferRecorder* recorder,
                         FrameId frame) {
    {
        std::unique_lock<std::mutex> lock(inputSamplingMutex);
        window->newImGuiFrame(frame);
        drawUI(frame);
        window->recordImGui(frame);
    }
    window->drawImGui(frame, recorder->get()->getCommandBuffer());
}

PerformanceCaptureData Engine::generatePerfCapture(
  FrameId lastReadyFrame, const FrameGraphSlops::ExtendedLatencyInfo& latency) const {
    FrameId firstFrame = lastReadyFrame > frameGraph.getMaxFramesInFlight() * 2 + 1
                           ? lastReadyFrame - (frameGraph.getMaxFramesInFlight() * 2 + 1)
                           : 0;
    uint32_t frameCount = uint32_t(lastReadyFrame - firstFrame + 1);
    drv::drv_assert(
      FrameGraph::TIMING_HISTORY_SIZE > frameCount + frameGraph.getMaxFramesInFlight(),
      "Timing history is not long enough to make a capture");
    FrameId targetFrame = lastReadyFrame - frameGraph.getMaxFramesInFlight();

    drv::drv_assert(latency.frame != INVALID_FRAME, "Latency frame is invalid");
    drv::drv_assert(targetFrame == latency.frame, "Latency frame != capture target frame");

    const FrameGraph::Node* inputSampler = frameGraph.getNode(inputSampleNode);
    const FrameGraph::Node* simStart =
      frameGraph.getNode(frameGraph.getStageStartNode(FrameGraph::SIMULATION_STAGE));
    const FrameGraph::Node* presentStart = frameGraph.getNode(presentFrameNode);

    FrameGraph::Node::NodeTiming firstPresentTiming =
      presentStart->getTiming(firstFrame, FrameGraph::RECORD_STAGE);
    FrameGraph::Node::NodeTiming targetTiming =
      simStart->getTiming(targetFrame, FrameGraph::SIMULATION_STAGE);
    FrameGraph::Node::NodeTiming inputSampleTiming =
      inputSampler->getTiming(targetFrame, FrameGraph::SIMULATION_STAGE);
    FrameGraph::Node::NodeTiming endPresentTiming =
      presentStart->getTiming(lastReadyFrame, FrameGraph::RECORD_STAGE);

    FrameGraph::Clock::time_point startTime = targetTiming.start;

    auto getTimeDiff = [](FrameGraph::Clock::time_point from, FrameGraph::Clock::time_point to) {
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count())
               / 1000000.0;
    };

    PerformanceCaptureData ret;
    ret.engineOptions = engineOptions;
    ret.frameId = targetFrame;
    ret.frameTime = getTimeDiff(firstPresentTiming.start, endPresentTiming.start) / frameCount;
    ret.fps = 1000.0 / ret.frameTime;
    ret.softwareLatency =
      double(std::chrono::nanoseconds(latency.info.inputNodeInfo.latencyNs).count()) / 1000000.0;
    ret.sleepTime =
      double(std::chrono::nanoseconds(inputSampleTiming.latencySleepNs).count()) / 1000000.0;
    ret.latencySlop =
      double(std::chrono::nanoseconds(latency.info.inputNodeInfo.totalSlopNs).count()) / 1000000.0;
    ret.workTime =
      double(std::chrono::nanoseconds(latency.frameLatencyInfo.asyncWorkNs).count()) / 1000000.0;
    ret.executionDelay = -1;
    ret.deviceDelay = -1;
    ret.frameEndFixPoint = getTimeDiff(startTime, frameEndFixPoint);

    for (FrameId frame = firstFrame; frame <= lastReadyFrame; ++frame) {
        const FrameGraph::FrameExecutionPackagesTimings& timing =
          frameGraph.getExecutionTiming(frame);
        double delay = 0;
        if (timing.packages.size() > 0)
            delay = double(timing.packages[timing.minDelay].delay.count()) / 1000.0;
        if (delay < 0)
            delay = 0;
        if (ret.executionDelay < 0 || delay < ret.executionDelay)
            ret.executionDelay = delay;
    }

    uint32_t pkgId = 0;

    struct NodeInfo
    {
        uint32_t pgkId = 0;
        std::string stageName = "";
        std::string threadName = "";
        uint32_t vecId = 0;
    };
    std::unordered_map<NodeId,
                       std::unordered_map<FrameId, std::unordered_map<FrameGraph::Stage, NodeInfo>>>
      cpuPackages;
    auto getPackageId = [&](NodeId node, FrameId frame, FrameGraph::Stage stage) {
        auto nodeItr = cpuPackages.find(node);
        if (nodeItr == cpuPackages.end())
            return NodeInfo{};
        auto frameItr = nodeItr->second.find(frame);
        if (frameItr == nodeItr->second.end())
            return NodeInfo{};
        auto stageItr = frameItr->second.find(stage);
        if (stageItr == frameItr->second.end())
            return NodeInfo{};
        return stageItr->second;
    };

    for (uint32_t stageId = 0; stageId < FrameGraph::NUM_STAGES; ++stageId) {
        FrameGraph::Stage stage = FrameGraph::get_stage(stageId);
        if (stage == FrameGraph::EXECUTION_STAGE)
            continue;
        std::string stageName = FrameGraph::get_stage_name(stage);
        ret.cpuStageOrder.push_back(stageName);
    }

    for (NodeId id = 0; id < frameGraph.getNodeCount(); ++id) {
        const FrameGraph::Node* node = frameGraph.getNode(id);
        for (uint32_t stageId = 0; stageId < FrameGraph::NUM_STAGES; ++stageId) {
            FrameGraph::Stage stage = FrameGraph::get_stage(stageId);
            if (stage == FrameGraph::EXECUTION_STAGE)
                continue;
            if (!node->hasStage(stage))
                continue;
            std::string stageName = FrameGraph::get_stage_name(stage);
            for (FrameId frame = firstFrame; frame <= lastReadyFrame; ++frame) {
                FrameGraph::Node::NodeTiming timing = node->getTiming(frame, stage);
                PerformanceCaptureCpuPackage pkg;
                pkg.name = node->getName();
                pkg.frameId = frame;
                pkg.packageId = pkgId++;
                pkg.slopDuration = double(timing.totalSlopNs) / 1000000.0;
                pkg.availableTime = getTimeDiff(startTime, timing.nodesReady);
                pkg.resAvailableTime = getTimeDiff(startTime, timing.resourceReady);
                pkg.startTime = getTimeDiff(startTime, timing.start);
                pkg.endTime = getTimeDiff(startTime, timing.finish);
                std::string threadName = get_thread_name(timing.threadId);
                cpuPackages[id][frame][stage] = {
                  pkg.packageId, stageName, threadName,
                  uint32_t(ret.stageToThreadToPackageList[stageName][threadName].size())};

                if (node->hasExecution() && stage == FrameGraph::RECORD_STAGE) {
                    FrameGraph::Node::ExecutionTiming execTiming = node->getExecutionTiming(frame);
                    PerformanceCaptureInterval interval;
                    interval.startTime = getTimeDiff(startTime, execTiming.start);
                    interval.endTime = getTimeDiff(startTime, execTiming.finish);
                    ret.executionIntervals[pkg.packageId] = std::move(interval);
                }
                ret.stageToThreadToPackageList[stageName][threadName].push_back(std::move(pkg));
            }
        }
        if (node->hasExecution())
            for (FrameId frame = firstFrame; frame <= lastReadyFrame; ++frame) {
                uint32_t submissionCount = node->getDeviceTimingCount(frame);
                for (uint32_t i = 0; i < submissionCount; ++i) {
                    FrameGraph::Node::DeviceTiming timing = node->getDeviceTiming(frame, i);
                    double delay = getTimeDiff(timing.submitted, timing.start);

                    // LOG_ENGINE("Device delay: %lf, submitted: %lf, start: %lf, finish: %lf", delay,
                    //            getTimeDiff(startTime, timing.submitted),
                    //            getTimeDiff(startTime, timing.start),
                    //            getTimeDiff(startTime, timing.finish));
                    if (delay < 0)
                        delay = 0;
                    if (ret.deviceDelay < 0 || delay < ret.deviceDelay)
                        ret.deviceDelay = delay;
                }
            }
    }
    for (NodeId id = 0; id < frameGraph.getNodeCount(); ++id) {
        const FrameGraph::Node* node = frameGraph.getNode(id);
        for (uint32_t stageId = 0; stageId < FrameGraph::NUM_STAGES; ++stageId) {
            FrameGraph::Stage stage = FrameGraph::get_stage(stageId);
            if (stage == FrameGraph::EXECUTION_STAGE)
                continue;
            std::string stageName = FrameGraph::get_stage_name(stage);
            if (!node->hasStage(stage))
                continue;
            for (FrameId frame = firstFrame; frame <= lastReadyFrame; ++frame) {
                NodeInfo current = getPackageId(id, frame, stage);
                if (current.stageName == "")
                    continue;
                for (const auto& dep : node->getCpuDeps()) {
                    if (dep.dstStage != stage)
                        continue;
                    if (frame < dep.offset)
                        continue;
                    if (dep.srcStage == FrameGraph::EXECUTION_STAGE) {
                        NodeInfo depended =
                          getPackageId(dep.srcNode, frame - dep.offset, FrameGraph::RECORD_STAGE);
                        if (depended.stageName != "") {
                            ret
                              .stageToThreadToPackageList[current.stageName][current.threadName]
                                                         [current.vecId]
                              .execDepended.insert(depended.pgkId);
                        }
                    }
                    NodeInfo depended = getPackageId(dep.srcNode, frame - dep.offset, dep.srcStage);
                    if (depended.stageName == "")
                        continue;
                    ret
                      .stageToThreadToPackageList[current.stageName][current.threadName]
                                                 [current.vecId]
                      .depended.insert(depended.pgkId);
                    ret
                      .stageToThreadToPackageList[depended.stageName][depended.threadName]
                                                 [depended.vecId]
                      .dependent.insert(current.pgkId);
                }
                for (const auto& dep : node->getGpuDeps()) {
                    NodeInfo dependedSource =
                      getPackageId(dep.srcNode, frame - dep.offset, FrameGraph::RECORD_STAGE);
                    if (dependedSource.stageName == "")
                        continue;
                    ret
                      .stageToThreadToPackageList[current.stageName][current.threadName]
                                                 [current.vecId]
                      .deviceDepended.insert(dependedSource.pgkId);
                }
                for (const auto& dep : node->getGpuDoneDeps()) {
                    if (dep.offset > frame)
                        continue;
                    ret
                      .stageToThreadToPackageList[current.stageName][current.threadName]
                                                 [current.vecId]
                      .gpuDoneDep[frameGraph.getQueueName(dep.srcQueue)] = frame - dep.offset;
                }
            }
        }
    }
    uint32_t execPgkId = 0;
    struct CmdBufferInfo
    {
        FrameId frameId;
        NodeId nodeId;
        drv::CmdBufferId cmdBufferId;
        bool operator<(const CmdBufferInfo& other) const {
            if (frameId != other.frameId)
                return frameId < other.frameId;
            if (nodeId != other.nodeId)
                return nodeId < other.nodeId;
            return cmdBufferId < other.cmdBufferId;
        }
    };
    std::map<CmdBufferInfo, uint32_t> cmdBufferToPkgId;
    for (FrameId frame = firstFrame; frame <= lastReadyFrame; ++frame) {
        const FrameGraph::FrameExecutionPackagesTimings& executionTimings =
          frameGraph.getExecutionTiming(frame);
        for (uint32_t i = 0; i < executionTimings.packages.size(); ++i) {
            NodeInfo sourcePackage = getPackageId(executionTimings.packages[i].sourceNode, frame,
                                                  FrameGraph::RECORD_STAGE);
            drv::drv_assert(sourcePackage.stageName != "", "Execution source node not found");
            PerformanceCaptureExecutionPackage package;
            // package.name = "<no_name>";  // TODO
            package.packageId = execPgkId++;
            package.sourcePackageId = sourcePackage.pgkId;
            package.issueTime = getTimeDiff(startTime, executionTimings.packages[i].submissionTime);
            package.startTime = getTimeDiff(startTime, executionTimings.packages[i].executionTime);
            package.endTime = getTimeDiff(startTime, executionTimings.packages[i].endTime);
            package.slopDuration = double(executionTimings.packages[i].totalSlopNs) / 1000000.0;
            package.minimalDelayInFrame = i == executionTimings.minDelay;
            if (executionTimings.packages[i].submissionId != drv::CmdBufferId(-1)) {
                CmdBufferInfo info{frame, executionTimings.packages[i].sourceNode,
                                   executionTimings.packages[i].submissionId};
                cmdBufferToPkgId[info] = package.packageId;
            }
            ret.executionPackages.push_back(std::move(package));
        }
    }
    for (NodeId id = 0; id < frameGraph.getNodeCount(); ++id) {
        const FrameGraph::Node* node = frameGraph.getNode(id);
        if (!node->hasExecution())
            continue;
        for (FrameId frame = firstFrame; frame <= lastReadyFrame; ++frame) {
            for (uint32_t i = 0; i < node->getDeviceTimingCount(frame); ++i) {
                FrameGraph::Node::DeviceTiming timing = node->getDeviceTiming(frame, i);
                CmdBufferInfo info{frame, id, timing.submissionId};
                auto execItr = cmdBufferToPkgId.find(info);
                drv::drv_assert(execItr != cmdBufferToPkgId.end(),
                                "Could not find submission source");
                PerformanceCaptureDevicePackage package;
                // package.name = ;
                package.sourceExecPackageId = execItr->second;
                package.slopDuration = double(timing.totalSlopNs) / 1000000.0;
                package.submissionTime = getTimeDiff(startTime, timing.submitted);
                package.startTime = getTimeDiff(startTime, timing.start);
                package.endTime = getTimeDiff(startTime, timing.finish);
                std::string queueName = frameGraph.getQueueName(timing.queueId);
                ret.queueToDevicePackageList[queueName].push_back(std::move(package));
            }
        }
    }
    return ret;
}

void Engine::createPerformanceCapture(FrameId targetFrame) {
    perfCaptureFrame = targetFrame;
    perFrameTempInfo[targetFrame % perFrameTempInfo.size()].captureHappening = true;
}

static void HelpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void Engine::drawUI(FrameId frameId) {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Quit", "Alt+F4")) {
                wantToQuit = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Options")) {
            ImGui::MenuItem("Latency", nullptr, &latencyOptionsOpen);
            ImGui::MenuItem("Workload", nullptr, &workloadOptionsOpen);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::BeginMenu("PerfMetrics")) {
                ImGui::Checkbox("Show perf metrics window", &engineOptions.perfMetrics_window);
                ImGui::Separator();
                ImGui::Checkbox("Fps", &engineOptions.perfMetrics_fps);
                ImGui::Checkbox("Theoretical fps", &engineOptions.perfMetrics_theoreticalFps);
                ImGui::Checkbox("Cpu work", &engineOptions.perfMetrics_cpuWork);
                ImGui::Checkbox("Exec queue work", &engineOptions.perfMetrics_execWork);
                ImGui::Checkbox("Device work", &engineOptions.perfMetrics_deviceWork);
                ImGui::Checkbox("Latency", &engineOptions.perfMetrics_latency);
                ImGui::Checkbox("Slop", &engineOptions.perfMetrics_slop);
                ImGui::Checkbox("Per frame slop", &engineOptions.perfMetrics_perFrameSlop);
                ImGui::Checkbox("Latency sleep", &engineOptions.perfMetrics_sleep);
                ImGui::Checkbox("Execution delay", &engineOptions.perfMetrics_execDelay);
                ImGui::Checkbox("Device delay", &engineOptions.perfMetrics_deviceDelay);
                ImGui::Checkbox("Work time", &engineOptions.perfMetrics_work);
                ImGui::Checkbox("Skipped or delayed frames",
                                &engineOptions.perfMetrics_skippedDelayed);

                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Game")) {
            recordMenuOptionsUI(frameId);
            ImGui::EndMenu();
        }

        float minCursorX = ImGui::GetCursorPosX();
        float cursorX =
          ImGui::GetCursorPosX() + ImGui::GetColumnWidth() - ImGui::GetStyle().ItemSpacing.x;
        cursorX -= ImGui::CalcTextSize("Reduce latency [x]").x;
        if (minCursorX <= cursorX) {
            ImGui::SetCursorPosX(cursorX);
            ImGui::Checkbox("Reduce latency", &engineOptions.latencyReduction);
        }
        if (latencyStats.hasInfo()) {
            cursorX -=
              ImGui::CalcTextSize("Latency: 999 (~99.9)").x + ImGui::GetStyle().ItemSpacing.x;
            if (minCursorX <= cursorX) {
                ImGui::SetCursorPosX(cursorX);
                ImGui::Text("Latency: %3.0lf (~%4.1lf)", latencyStats.getAvg(),
                            latencyStats.getStdDiv());
            }
        }
        if (fpsStats.hasInfo()) {
            cursorX -= ImGui::CalcTextSize("Fps: 999 (~99.9)").x + ImGui::GetStyle().ItemSpacing.x;
            if (minCursorX <= cursorX) {
                ImGui::SetCursorPosX(cursorX);
                ImGui::Text("Fps: %3.0lf (~%4.1lf)", fpsStats.getAvg(), fpsStats.getStdDiv());
            }
        }
        ImGui::EndMainMenuBar();
    }

    if (engineOptions.perfMetrics_window) {
        ImGui::Begin("Performance metrics", &engineOptions.perfMetrics_window, 0);
        if (engineOptions.perfMetrics_fps && fpsStats.hasInfo())
            ImGui::Text("Fps:             %3.0lf (~%4.1lf)", fpsStats.getAvg(),
                        fpsStats.getStdDiv());
        if (engineOptions.perfMetrics_theoreticalFps && theoreticalFpsStats.hasInfo())
            ImGui::Text("Theoretical fps: %3.0lf (~%4.1lf)", theoreticalFpsStats.getAvg(),
                        theoreticalFpsStats.getStdDiv());
        if (engineOptions.perfMetrics_cpuWork && cpuWorkStats.hasInfo() && cpuOffsetStats.hasInfo())
            ImGui::Text("Cpu work:        %3.0lf (~%4.1lf) -> %3.0lf fps", cpuWorkStats.getAvg(),
                        cpuWorkStats.getStdDiv(), (1000.0 / cpuOffsetStats.getAvg()));
        if (engineOptions.perfMetrics_execWork && execWorkStats.hasInfo()
            && execOffsetStats.hasInfo())
            ImGui::Text("Exec work:       %3.0lf (~%4.1lf) -> %3.0lf fps", execWorkStats.getAvg(),
                        execWorkStats.getStdDiv(), (1000.0 / execOffsetStats.getAvg()));
        if (engineOptions.perfMetrics_deviceWork && deviceWorkStats.hasInfo()
            && deviceOffsetStats.hasInfo())
            ImGui::Text("Device work:     %3.0lf (~%4.1lf) -> %3.0lf fps", deviceWorkStats.getAvg(),
                        deviceWorkStats.getStdDiv(), (1000.0 / deviceOffsetStats.getAvg()));
        if (engineOptions.perfMetrics_latency && latencyStats.hasInfo())
            ImGui::Text("Latency:         %3.0lf (~%4.1lf)", latencyStats.getAvg(),
                        latencyStats.getStdDiv());
        if (engineOptions.perfMetrics_slop && slopStats.hasInfo())
            ImGui::Text("Slop:            %3.0lf (~%4.1lf)", slopStats.getAvg(),
                        slopStats.getStdDiv());
        if (engineOptions.perfMetrics_perFrameSlop && perFrameSlopStats.hasInfo())
            ImGui::Text("Per frame slop:  %3.0lf (~%4.1lf)", perFrameSlopStats.getAvg(),
                        perFrameSlopStats.getStdDiv());
        if (engineOptions.perfMetrics_work && workStats.hasInfo())
            ImGui::Text("Work time:       %3.0lf (~%4.1lf)", workStats.getAvg(),
                        workStats.getStdDiv());
        if (engineOptions.perfMetrics_sleep && waitTimeStats.hasInfo())
            ImGui::Text("Sleep:           %3.0f (~%4.1lf)", waitTimeStats.getAvg(),
                        waitTimeStats.getStdDiv());
        if (engineOptions.perfMetrics_execDelay && execDelayStats.hasInfo())
            ImGui::Text("Execution delay: %3.0f (~%4.1lf)", execDelayStats.getAvg(),
                        execDelayStats.getStdDiv());
        if (engineOptions.perfMetrics_deviceDelay && deviceDelayStats.hasInfo())
            ImGui::Text("Device delay:    %3.0f (~%4.1lf)", deviceDelayStats.getAvg(),
                        deviceDelayStats.getStdDiv());
        if (engineOptions.perfMetrics_skippedDelayed && skippedDelayed.hasInfo()
            && skippedDelayed.getAvg() >= 0)
            ImGui::Text("Skipped frames:  %5.1lf%%", skippedDelayed.getAvg() * 100.0);

        ImGui::End();
    }

    if (latencyOptionsOpen) {
        ImGui::Begin("Latency options", &latencyOptionsOpen, 0);
        ImGui::Checkbox("Latency reduction    ", &engineOptions.latencyReduction);
        {
            if (!engineOptions.latencyReduction)
                ImGui::BeginDisabled();
            const char* fpsModes[] = {"unlimited", "limited", "discretized"};
            int currentMode = static_cast<int>(engineOptions.refreshMode);
            ImGui::Combo("Mode", &currentMode, fpsModes, IM_ARRAYSIZE(fpsModes));
            engineOptions.refreshMode = static_cast<EngineOptions::RefreshRateMode>(currentMode);
            ImGui::DragFloat("Target fps", &engineOptions.targetRefreshRate, 0.5, 1.f, 1000.f,
                             "%.8f fps");
            ImGui::DragFloat("Desired slop", &engineOptions.desiredSlop, 0.1f, 0, 32, "%.8fms");
            // ImGui::DragFloat("Work prediction", &engineOptions.workPrediction, 0.05f, 0, 10,
            //                  "avg + %.8f*stdDiv");
            ImGui::Checkbox("Manual latency sleep ", &engineOptions.manualLatencyReduction);
            ImGui::DragFloat("Manual sleep time", &engineOptions.manualSleepTime, 0.1f, 0, 100,
                             "%.8fms");
            ImGui::DragFloat("Work time smoothing", &engineOptions.workTimeSmoothing, 0.05f, 0.f,
                             0.95f, "lerp(cur, avg, %.8f)");
            if (!engineOptions.latencyReduction)
                ImGui::EndDisabled();
        }

        ImGui::End();
    }

    if (workloadOptionsOpen) {
        ImGui::Begin("Workload options", &workloadOptionsOpen, 0);
        ImGui::DragFloat("Before input avg", &engineOptions.manualWorkload_beforeInputAvg, 0.5, 0.f,
                         100.f, "%.8f ms");
        ImGui::DragFloat("Before input std div", &engineOptions.manualWorkload_beforeInputStdDiv,
                         0.25, 0.f, 50.f, "~%.8f ms");
        ImGui::DragFloat("After input avg", &engineOptions.manualWorkload_afterInputAvg, 0.5, 0.f,
                         100.f, "%.8f ms");
        ImGui::DragFloat("After input std div", &engineOptions.manualWorkload_afterInputStdDiv,
                         0.25, 0.f, 50.f, "~%.8f ms");
        ImGui::DragFloat("Exec avg", &engineOptions.manualWorkload_execInputAvg, 0.5, 0.f, 100.f,
                         "%.8f ms");
        ImGui::DragFloat("Exec std div", &engineOptions.manualWorkload_execInputStdDiv, 0.25, 0.f,
                         50.f, "~%.8f ms");
        ImGui::DragFloat("Device avg", &engineOptions.manualWorkload_deviceInputAvg, 1.f, 0.f,
                         10000.f, "%.8f iterations");
        ImGui::DragFloat("Device std div", &engineOptions.manualWorkload_deviceInputStdDiv, 1.f,
                         0.f, 5000.f, "~%.8f iterations");

        ImGui::End();
    }

    recordGameUI(frameId);
}

void TransformRecord::interpolate(float time, glm::quat& orientation, glm::vec3& position) const {
    drv::drv_assert(!entries.empty(), "TransfromRecord object is empty");
    if (time <= entries[0].timeMs) {
        orientation = entries[0].orientation;
        position = entries[0].position;
    }
    else if (time >= entries.back().timeMs) {
        orientation = entries.back().orientation;
        position = entries.back().position;
    }
    else {
        uint32_t a = 0;
        uint32_t b = uint32_t(entries.size());
        while (a + 1 < b) {
            uint32_t m = (a + b) / 2;
            if (time < entries[m].timeMs)
                b = m;
            else
                a = m;
        }
        if (b == entries.size()) {
            orientation = entries[a].orientation;
            position = entries[a].position;
        }
        else {
            float p = (time - entries[a].timeMs) / (entries[b].timeMs - entries[a].timeMs);
            position = lerp(entries[a].position, entries[b].position, p);
            orientation = glm::slerp(entries[a].orientation, entries[b].orientation, p);
        }
    }
}
