#include "game.h"

// #include <iostream>
#include <optional>

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

    testRenderPass = drv::create_render_pass(engine->getDevice(), "Test pass");
    drv::RenderPass::AttachmentInfo colorInfo;
    testColorAttachment = testRenderPass->createAttachment(std::move(colorInfo));
    drv::RenderPass::SubpassInfo subpassInfo;
    testSubpass = testRenderPass->createSubpass(std::move(subpassInfo));
    testRenderPass->build();
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

drv::ImageViewPtr Game::getView(drv::ImagePtr image, uint32_t imageIndex) {
    while (imageViews.size() <= imageIndex)
        imageViews.emplace_back(
          this,
          [](const Game*, drv::ImagePtr current, drv::ImagePtr newParent, const drv::ImageView&) {
              // eq
              return current == newParent;
          },
          [](const Game* game, drv::ImagePtr parent) {
              // gen
              drv::ImageViewCreateInfo createInfo;
              createInfo.image = parent;
              createInfo.type = drv::ImageViewCreateInfo::TYPE_2D;
              createInfo.format = drv::get_texture_info(parent).format;
              createInfo.components.r = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
              createInfo.components.g = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
              createInfo.components.b = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
              createInfo.components.a = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
              createInfo.subresourceRange.aspectMask = drv::COLOR_BIT;
              createInfo.subresourceRange.baseArrayLayer = 0;
              createInfo.subresourceRange.baseMipLevel = 0;
              createInfo.subresourceRange.layerCount =
                createInfo.subresourceRange.REMAINING_ARRAY_LAYERS;
              createInfo.subresourceRange.levelCount =
                createInfo.subresourceRange.REMAINING_MIP_LEVELS;
              return drv::ImageView(game->engine->getDevice(), createInfo);
          },
          [](const Game* game, drv::ImageView&& view) {
              // del
              game->engine->getGarbageSystem()->useGarbage(
                [&](Garbage* trashBin) { trashBin->releaseImageView(std::move(view)); });
          });
    return imageViews[imageIndex].bind(image);
}

void Game::record(FrameGraph& frameGraph, FrameId frameId) {
    std::cout << "Record: " << frameId << std::endl;
    Engine::QueueInfo queues = engine->getQueues();
    if (FrameGraph::NodeHandle testDrawHandle = frameGraph.acquireNode(testDraw, frameId);
        testDrawHandle) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        Engine::AcquiredImageData swapChainData = engine->acquiredSwapchainImage(testDrawHandle);
        Engine::CommandBufferRecorder recorder =
          engine->acquireCommandRecorder(testDrawHandle, frameId, queues.renderQueue.id);
        if (frameId < 3)
            recorder.getResourceTracker()->enableCommandLog();
        drv::RenderPass::ImageInfo testImageInfo[1];
        testImageInfo[testColorAttachment].image = swapChainData.image;
        testImageInfo[testColorAttachment].view =
          getView(swapChainData.image, swapChainData.imageIndex);
        drv::CmdRenderPass testPass = testRenderPass->begin(testImageInfo);
        testPass.beginSubpass(testSubpass);
        testPass.end();

        /// --- oroginal clear ---
        drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
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
        /// --- clear ---

        // TODO according to vulkan spec https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueuePresentKHR.html
        // memory is made visible to all read operations (add this to tracker?) -- only available memory
        recorder.cmdSignalSemaphore(swapChainData.renderFinishedSemaphore);
        recorder.finishQueueWork();
        if (frameId > 3)
            recorder.getResourceTracker()->disableCommandLog();
    }
    else
        assert(frameGraph.isStopped());
}

void Game::simulate(FrameGraph& frameGraph, FrameId frameId) {
    UNUSED(frameGraph);
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    std::cout << "Simulate: " << frameId << std::endl;
}
