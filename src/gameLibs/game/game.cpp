#include "game.h"

// #include <iostream>

#include <engine.h>

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
                                FrameGraph::NodeId& presentDepNode, drv::QueuePtr& depQueue) {
    testDraw = frameGraph.addNode(FrameGraph::Node("testDraw", true));
    frameGraph.addDependency(testDraw, FrameGraph::CpuDependency{data.recStart, 0});
    frameGraph.addDependency(testDraw, FrameGraph::EnqueueDependency{data.recStart, 0});
    frameGraph.addDependency(data.recEnd, FrameGraph::CpuDependency{testDraw, 0});
    frameGraph.addDependency(data.recEnd, FrameGraph::EnqueueDependency{testDraw, 0});

    presentDepNode = testDraw;
    depQueue = engine->getQueues().renderQueue;
    return true;
}

void Game::initSimulationFrameGraph(FrameGraph& frameGraph,
                                    const ISimulation::FrameGraphData& data) {
    // TODO
}

void Game::record(FrameGraph& frameGraph, FrameGraph::FrameId frameId) {
    std::cout << "Record: " << frameId << std::endl;
    Engine::QueueInfo queues = engine->getQueues();
    FrameGraph::NodeHandle testDrawHandle = frameGraph.acquireNode(testDraw, frameId);
    if (testDrawHandle) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        // Engine::AcquiredImageData engine->acquiredSwapchainImage(testDrawHandle); // TODO
        Engine::CommandBufferRecorder recorder =
          engine->acquireCommandRecorder(testDrawHandle, frameId, queues.renderQueue);
    }
    else
        assert(frameGraph.isStopped());
}

void Game::simulate(FrameGraph& frameGraph, FrameGraph::FrameId frameId) {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    std::cout << "Simulate: " << frameId << std::endl;
}
