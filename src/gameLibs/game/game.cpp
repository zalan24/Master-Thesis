#include "game.h"

#include <optional>

#include <util.hpp>

#include <drverror.h>

#include <garbage.h>

#include <shader_obj_test.h>

Game::Game(Engine* _engine)
  : engine(_engine),
    shaderHeaders(engine->getDevice()),
    shaderObjects(engine->getDevice(), *engine->getShaderBin()) {
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
    colorInfo.initialLayout = drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    colorInfo.finalLayout = drv::ImageLayout::PRESENT_SRC_KHR;
    colorInfo.loadOp = drv::AttachmentLoadOp::CLEAR;
    colorInfo.storeOp = drv::AttachmentStoreOp::STORE;
    colorInfo.stencilLoadOp = drv::AttachmentLoadOp::DONT_CARE;
    colorInfo.stencilStoreOp = drv::AttachmentStoreOp::DONT_CARE;
    // colorInfo.srcUsage = 0;
    // colorInfo.dstUsage = drv::IMAGE_USAGE_PRESENT;
    testColorAttachment = testRenderPass->createAttachment(std::move(colorInfo));
    drv::SubpassInfo subpassInfo;
    subpassInfo.colorOutputs.push_back(
      {testColorAttachment, drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL});
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

void Game::recreateViews(uint32_t imageCount, const drv::ImagePtr* images) {
    frameBuffers.clear();
    while (imageViews.size()) {
        engine->getGarbageSystem()->useGarbage([this](Garbage* trashBin) {
            trashBin->releaseImageView(std::move(imageViews.back()));
            imageViews.pop_back();
        });
    }
    for (uint32_t i = 0; i < imageCount; ++i) {
        drv::ImageViewCreateInfo createInfo;
        createInfo.image = images[i];
        createInfo.type = drv::ImageViewCreateInfo::TYPE_2D;
        createInfo.format = drv::get_texture_info(images[i]).format;
        createInfo.components.r = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.g = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.b = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.a = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.subresourceRange.aspectMask = drv::COLOR_BIT;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.layerCount = createInfo.subresourceRange.REMAINING_ARRAY_LAYERS;
        createInfo.subresourceRange.levelCount = createInfo.subresourceRange.REMAINING_MIP_LEVELS;
        imageViews.emplace_back(engine->getDevice(), createInfo);
        frameBuffers.emplace_back(engine->getDevice());
    }
}

void Game::record(FrameGraph& frameGraph, FrameId frameId) {
    // std::cout << "Record: " << frameId << std::endl;
    Engine::QueueInfo queues = engine->getQueues();
    if (FrameGraph::NodeHandle testDrawHandle = frameGraph.acquireNode(testDraw, frameId);
        testDrawHandle) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        Engine::AcquiredImageData swapChainData = engine->acquiredSwapchainImage(testDrawHandle);
        drv::drv_assert(swapChainData.version != Engine::INVALID_SWAPCHAIN, "Handle this somehow");
        if (swapchainVersion != swapChainData.version) {
            recreateViews(swapChainData.imageCount, swapChainData.images);
            swapchainVersion = swapChainData.version;
        }
        // LOG_F(INFO, "Frame %lld  Swapchain image: %d Image: %p", frameId, swapChainData.imageIndex,
        //       static_cast<const void*>(swapChainData.image));
        Engine::CommandBufferRecorder recorder =
          engine->acquireCommandRecorder(testDrawHandle, frameId, queues.renderQueue.id);
        if (frameId < 3)
            recorder.getResourceTracker()->enableCommandLog();
        recorder.cmdWaitSemaphore(
          swapChainData.imageAvailableSemaphore,
          drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE | drv::IMAGE_USAGE_TRANSFER_DESTINATION);

        recorder.cmdImageBarrier(
          {swapChainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
           drv::ImageMemoryBarrier::AUTO_TRANSITION, true,
           drv::get_queue_family(engine->getDevice(), queues.renderQueue.handle)});
        drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
        if ((frameId / 100) % 2 == 0)
            clearValue = drv::ClearColorValue(1.f, 1.f, 0.f, 1.f);
        else
            clearValue = drv::ClearColorValue(0.f, 1.f, 1.f, 1.f);
        recorder.cmdClearImage(swapChainData.image, &clearValue);

        recorder.cmdImageBarrier(
          {swapChainData.image, drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE,
           drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false,
           drv::get_queue_family(engine->getDevice(), queues.renderQueue.handle)});
        drv::RenderPass::AttachmentData testImageInfo[1];
        testImageInfo[testColorAttachment].image = swapChainData.image;
        testImageInfo[testColorAttachment].view = imageViews[swapChainData.imageIndex];
        if (testRenderPass->needRecreation(testImageInfo)) {
            for (auto& framebuffer : frameBuffers)
                framebuffer.reset();
            testRenderPass->recreate(testImageInfo);
        }
        if (!frameBuffers[swapChainData.imageIndex])
            frameBuffers[swapChainData.imageIndex].set(
              testRenderPass->createFramebuffer(testImageInfo));
        drv::ClearValue clearValues[1];
        clearValues[testColorAttachment].type = drv::ClearValue::COLOR;
        if ((frameId / 100) % 2 == 0)
            clearValues[testColorAttachment].value.color = drv::ClearColorValue(0.f, 1.f, 0.f, 1.f);
        else
            clearValues[testColorAttachment].value.color = drv::ClearColorValue(1.f, 0.f, 0.f, 1.f);
        // clearValues[testColorAttachment].value.color = drv::ClearColorValue(255, 255, 255, 255);
        drv::Rect2D renderArea;
        renderArea.extent = swapChainData.extent;
        renderArea.offset = {0, 0};
        drv::CmdRenderPass testPass =
          testRenderPass->begin(recorder.getResourceTracker(), recorder.getCommandBuffer(),
                                frameBuffers[swapChainData.imageIndex], renderArea, clearValues);
        testPass.beginSubpass(testSubpass);
        drv::ClearRect clearRect;
        clearRect.rect.offset = {50, 50};
        clearRect.rect.extent = {swapChainData.extent.width - 100,
                                 swapChainData.extent.height - 100};
        clearRect.baseLayer = 0;
        clearRect.layerCount = 1;
        testPass.clearColorAttachment(testColorAttachment, drv::ClearColorValue(0.f, 0.f, 1.f, 1.f),
                                      1, &clearRect);
        testPass.end();

        // /// --- oroginal clear ---
        // recorder.cmdImageBarrier(
        //   {swapChainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
        //    drv::ImageMemoryBarrier::AUTO_TRANSITION, true,
        //    drv::get_queue_family(engine->getDevice(), queues.renderQueue.handle)});
        // drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
        // recorder.cmdClearImage(swapChainData.image, &clearValue);
        // /// --- clear ---

        recorder.cmdImageBarrier(
          {swapChainData.image, drv::IMAGE_USAGE_PRESENT, drv::ImageMemoryBarrier::AUTO_TRANSITION,
           false, drv::get_queue_family(engine->getDevice(), queues.presentQueue.handle)});
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
    UNUSED(frameId);
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    // std::cout << "Simulate: " << frameId << std::endl;
}
