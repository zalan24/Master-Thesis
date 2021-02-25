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

void Game::initRenderFrameGraph(FrameGraph& frameGraph, const IRenderer::FrameGraphData& data) {
    // TODO
}

void Game::initSimulationFrameGraph(FrameGraph& frameGraph,
                                    const ISimulation::FrameGraphData& data) {
    // TODO
}

void Game::record(FrameGraph::FrameId frameId) {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    // std::cout << "Simulation: " << frameId << std::endl;
}

void Game::simulate(FrameGraph::FrameId frameId) {
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    // std::cout << "Record: " << frameId << std::endl;
}
