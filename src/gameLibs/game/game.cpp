#include "game.h"

// #include <iostream>

#include <util.hpp>

#include <engine.h>
#include <garbage.h>

#include <shader_obj_test.h>

Game::Game(Engine* _engine) : engine(_engine) {
    shader_obj_test testShader(engine->getDevice(), *engine->getShaderBin());
    shader_obj_test::Descriptor descriptor;
    descriptor.setVariant("Color", "red");
    descriptor.setVariant("TestVariant", "two");
    descriptor.desc_global.setVariant_renderPass(shader_global_descriptor::Renderpass::DEPTH);
    descriptor.desc_global.setVariant_someStuff(shader_global_descriptor::Somestuff::STUFF3);
    std::cout << descriptor.desc_global.getLocalVariantId() << std::endl;
    std::cout << descriptor.desc_test.getLocalVariantId() << std::endl;
    std::cout << descriptor.getLocalVariantId() << std::endl;
}

Game::~Game() {
}

bool Game::initRenderFrameGraph(FrameGraph& frameGraph, const IRenderer::FrameGraphData& data,
                                FrameGraph::NodeId& presentDepNode,
                                FrameGraph::QueueId& depQueueId) {
    testDraw = frameGraph.addNode(FrameGraph::Node("testDraw", true));
    frameGraph.addDependency(testDraw, FrameGraph::CpuDependency{data.recStart, 0});
    frameGraph.addDependency(testDraw, FrameGraph::EnqueueDependency{data.recStart, 0});
    frameGraph.addDependency(data.recEnd, FrameGraph::CpuDependency{testDraw, 0});
    frameGraph.addDependency(data.recEnd, FrameGraph::EnqueueDependency{testDraw, 0});

    presentDepNode = testDraw;
    depQueueId = engine->getQueues().renderQueue.id;
    return true;
}

void Game::initSimulationFrameGraph(FrameGraph& frameGraph,
                                    const ISimulation::FrameGraphData& data) {
    UNUSED(frameGraph);
    UNUSED(data);
    // TODO
}

void Game::record(FrameGraph& frameGraph, FrameId frameId) {
    std::cout << "Record: " << frameId << std::endl;
    Engine::QueueInfo queues = engine->getQueues();
    FrameGraph::NodeHandle testDrawHandle = frameGraph.acquireNode(testDraw, frameId);
    if (testDrawHandle) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        Engine::AcquiredImageData swapChainData = engine->acquiredSwapchainImage(testDrawHandle);
        Engine::CommandBufferRecorder recorder =
          engine->acquireCommandRecorder(testDrawHandle, frameId, queues.renderQueue.id);
        drv::ClearColorValue clearValue(255u, 255u, 0u, 255u);
        recorder.cmdWaitSemaphore(swapChainData.imageAvailableSemaphore,
                                  drv::IMAGE_USAGE_TRANSFER_DESTINATION);
        recorder.cmdImageBarrier(
          {swapChainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
           drv::ImageMemoryBarrier::AUTO_TRANSITION, true,
           drv::get_queue_family(engine->getDevice(), queues.renderQueue.handle)});
        recorder.cmdClearImage(swapChainData.image, &clearValue);
        recorder.cmdImageBarrier(
          {swapChainData.image, drv::IMAGE_USAGE_PRESENT, drv::ImageMemoryBarrier::AUTO_TRANSITION,
           false, drv::get_queue_family(engine->getDevice(), queues.presentQueue.handle)});
        // TODO according to vulkan spec https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueuePresentKHR.html
        // memory is made visible to all read operations (add this to tracker?) -- only available memory
        recorder.cmdSignalSemaphore(swapChainData.renderFinishedSemaphore);
        // recorder.finishQueueWork();
    }
    else
        assert(frameGraph.isStopped());
}

void Game::simulate(FrameGraph& frameGraph, FrameId frameId) {
    UNUSED(frameGraph);
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    std::cout << "Simulate: " << frameId << std::endl;
}
