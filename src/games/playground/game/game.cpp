#include "game.h"

#include <optional>

#include <util.hpp>

#include <drverror.h>
#include <renderpass.h>

Game::Game(int argc, char* argv[], const Config& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           ResourceManager::ResourceInfos resource_infos, const Args& args)
  : Game3D(argc, argv, config, trackingConfig, shaderbinFile, resource_infos, args),
    shaderHeaders(getDevice()),
    shaderObjects(getDevice(), *getShaderBin(), shaderHeaders),
    shaderGlobalDesc(getDevice(), &shaderHeaders.global),
    shaderTestDesc(getDevice(), &shaderHeaders.test),
    dynamicStates(drv::DrvShader::DynamicStates::FIXED_SCISSOR,
                  drv::DrvShader::DynamicStates::FIXED_VIEWPORT),
    testShader(getDevice(), &shaderObjects.test, dynamicStates) {
    // shader_obj_test::Descriptor descriptor;
    // descriptor.setVariant("Color", "red");
    // descriptor.setVariant("TestVariant", "two");
    // descriptor.desc_global.setVariant_renderPass(shader_global_descriptor::Renderpass::DEPTH);
    // descriptor.desc_global.setVariant_someStuff(shader_global_descriptor::Somestuff::STUFF3);
    // std::cout << descriptor.desc_global.getLocalVariantId() << std::endl;
    // std::cout << descriptor.desc_test.getLocalVariantId() << std::endl;
    // std::cout << descriptor.getLocalVariantId() << std::endl;

    testRenderPass = drv::create_render_pass(getDevice(), "Test pass");
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

    testDraw = getFrameGraph().addNode(
      FrameGraph::Node("testDraw", FrameGraph::RECORD_STAGE | FrameGraph::EXECUTION_STAGE));

    // TODO present node could be inside of record end node???
    buildFrameGraph(testDraw, getQueues().renderQueue.id);
}

Game::~Game() {
}

static ShaderObject::DynamicState get_dynamic_states(drv::Extent2D extent) {
    ShaderObject::DynamicState ret;
    ret.scissor.offset = {0, 0};
    ret.scissor.extent = extent;
    ret.viewport.x = 0;
    ret.viewport.y = 0;
    ret.viewport.width = static_cast<float>(extent.width);
    ret.viewport.height = static_cast<float>(extent.height);
    ret.viewport.minDepth = 0;
    ret.viewport.maxDepth = 1;
    return ret;
}

void Game::record_cmd_buffer(const RecordData& data, drv::DrvCmdBufferRecorder* recorder) {
    recorder->registerUndefinedImage(data.targetImage);

    recorder->cmdImageBarrier({data.targetImage, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, true,
                               drv::get_queue_family(data.device, data.renderQueue)});
    drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
    if (data.variant == 0) {
        clearValue = drv::ClearColorValue(1.f, 1.f, 0.f, 1.f);
        data.shaderTestDesc->setVariant_Color(shader_test_descriptor::Color::BLUE);
    }
    else {
        clearValue = drv::ClearColorValue(0.f, 1.f, 1.f, 1.f);
        data.shaderTestDesc->setVariant_Color(shader_test_descriptor::Color::RED);
    }

    recorder->cmdClearImage(data.targetImage, &clearValue);

    recorder->cmdImageBarrier({data.targetImage, drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE,
                               drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false,
                               drv::get_queue_family(data.device, data.renderQueue)});
    drv::ClearValue clearValues[1];
    clearValues[data.testColorAttachment].type = drv::ClearValue::COLOR;
    if (data.variant == 0)
        clearValues[data.testColorAttachment].value.color =
          drv::ClearColorValue(0.1f, 0.8f, 0.1f, 1.f);
    else
        clearValues[data.testColorAttachment].value.color =
          drv::ClearColorValue(0.8f, 0.1f, 0.1f, 1.f);
    // clearValues[data.testColorAttachment].value.color = drv::ClearColorValue(255, 255, 255, 255);
    drv::Rect2D renderArea;
    renderArea.extent = data.extent;
    renderArea.offset = {0, 0};
    EngineRenderPass testPass(data.renderPass, recorder, renderArea, data.frameBuffer, clearValues);
    testPass.beginSubpass(data.testSubpass);
    drv::ClearRect clearRect;
    clearRect.rect.offset = {100, 100};
    clearRect.rect.extent = {data.extent.width - 200, data.extent.height - 200};
    clearRect.baseLayer = 0;
    clearRect.layerCount = 1;
    testPass.clearColorAttachment(data.testColorAttachment,
                                  drv::ClearColorValue(0.f, 0.7f, 0.7f, 1.f), 1, &clearRect);
    testPass.bindGraphicsShader(get_dynamic_states(data.extent), {}, *data.testShader,
                                data.shaderGlobalDesc, data.shaderTestDesc);
    // testShader.bindGraphicsInfo(ShaderObject::NORMAL_USAGE, testPass,
    //                             get_dynamic_states(swapChainData.extent), &shaderGlobalDesc,
    //                             &data.shaderTestDesc);
    testPass.draw(3, 1, 0, 0);
    testPass.end();

    // /// --- oroginal clear ---
    // recorder->cmdImageBarrier(
    //   {swapChainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
    //    drv::ImageMemoryBarrier::AUTO_TRANSITION, true,
    //    drv::get_queue_family(data.device, queues.renderQueue.handle)});
    // drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
    // recorder->cmdClearImage(swapChainData.image, &clearValue);
    // /// --- clear ---

    recorder->cmdImageBarrier({data.targetImage, drv::IMAGE_USAGE_PRESENT,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, false,
                               drv::get_queue_family(data.device, data.presentQueue)});
    // TODO according to vulkan spec https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueuePresentKHR.html
    // memory is made visible to all read operations (add this to tracker?) -- only available memory
}

Engine::AcquiredImageData Game::record(FrameId frameId) {
    // std::cout << "Record: " << frameId << std::endl;
    RUNTIME_STAT_SCOPE(gameRecord);
    Engine::QueueInfo queues = getQueues();
    if (FrameGraph::NodeHandle testDrawHandle =
          getFrameGraph().acquireNode(testDraw, FrameGraph::RECORD_STAGE, frameId);
        testDrawHandle) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        Engine::AcquiredImageData swapChainData = acquiredSwapchainImage(testDrawHandle);
        if (!swapChainData)
            return swapChainData;
        // LOG_F(INFO, "Frame %lld  Swapchain image: %d Image: %p", frameId, swapChainData.imageIndex,
        //       static_cast<const void*>(swapChainData.image));

        // drv::RenderPass::AttachmentData testImageInfo[1];
        // testImageInfo[testColorAttachment].image = swapChainData.image;
        // testImageInfo[testColorAttachment].view = imageViews[swapChainData.imageIndex].get();
        // if (testRenderPass->needRecreation(testImageInfo)) {
        //     for (auto& framebuffer : frameBuffers)
        //         framebuffer = {};
        //     testRenderPass->recreate(testImageInfo);
        //     initShader(swapChainData.extent);
        // }
        // if (!frameBuffers[swapChainData.imageIndex]
        //     || !frameBuffers[swapChainData.imageIndex].get())
        //     frameBuffers[swapChainData.imageIndex] = createResource<drv::Framebuffer>(
        //       getDevice(), testRenderPass->createFramebuffer(testImageInfo));
        testRenderPass->attach(&swapchainAttachments[swapChainData.imageIndex]);

        OneTimeCmdBuffer<RecordData> cmdBuffer(getPhysicalDevice(), getDevice(),
                                               queues.renderQueue.handle, getCommandBufferBank(),
                                               getGarbageSystem(), record_cmd_buffer);
        RecordData recordData;
        recordData.device = getDevice();
        recordData.targetImage = swapChainData.image;
        recordData.targetView = imageViews[swapChainData.imageIndex].get();
        recordData.testColorAttachment = testColorAttachment;
        recordData.variant = (frameId / 100) % 2;
        recordData.extent = swapChainData.extent;
        recordData.renderQueue = queues.renderQueue.handle;
        recordData.presentQueue = queues.presentQueue.handle;
        recordData.renderPass = testRenderPass.get();
        recordData.testShader = &testShader;
        recordData.shaderTestDesc = &shaderTestDesc;
        recordData.frameBuffer = swapchainFrameBuffers[swapChainData.imageIndex].get();
        recordData.testSubpass = testSubpass;
        recordData.shaderGlobalDesc = &shaderGlobalDesc;
        ExecutionPackage::CommandBufferPackage submission = make_submission_package(
          queues.renderQueue.handle, cmdBuffer.use(std::move(recordData)), getGarbageSystem(),
          ResourceStateValidationMode::IGNORE_FIRST_SUBMISSION);
        submission.signalSemaphores.push_back(swapChainData.renderFinishedSemaphore);
        submission.waitSemaphores.push_back(
          {swapChainData.imageAvailableSemaphore,
           drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE | drv::IMAGE_USAGE_TRANSFER_DESTINATION});
        testDrawHandle.submit(queues.renderQueue.id, std::move(submission));
        //   acquireCommandRecorder(testDrawHandle, frameId, queues.renderQueue.id);
        return swapChainData;
    }
    return {};
}

void Game::simulate(FrameId frameId) {
    UNUSED(frameId);
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    // std::cout << "Simulate: " << frameId << std::endl;
}

void Game::beforeDraw(FrameId) {
}

void Game::readback(FrameId) {
}

void Game::releaseSwapchainResources() {
    getGarbageSystem()->useGarbage([this](Garbage* trashBin) { testShader.clear(trashBin); });
    swapchainFrameBuffers.clear();
    swapchainAttachments.clear();
    imageViews.clear();
}

void Game::createSwapchainResources(const drv::Swapchain& swapchain) {
    imageViews.reserve(swapchain.getImageCount());
    swapchainAttachments.reserve(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); ++i) {
        drv::ImageViewCreateInfo createInfo;
        createInfo.image = swapchain.getImages()[i];
        createInfo.type = drv::ImageViewCreateInfo::TYPE_2D;
        createInfo.format = drv::get_texture_info(swapchain.getImages()[i]).format;
        createInfo.components.r = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.g = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.b = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.a = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.subresourceRange.aspectMask = drv::COLOR_BIT;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.layerCount = 1;
        createInfo.subresourceRange.levelCount = 1;
        imageViews.push_back(createResource<drv::ImageView>(getDevice(), createInfo));

        drv::RenderPass::AttachmentData swapchainAttachment;
        swapchainAttachment.image = swapchain.getImages()[i];
        swapchainAttachment.view = imageViews.back().get();
        swapchainAttachments.push_back(std::move(swapchainAttachment));
    }
    if (!testRenderPass->isCompatible(&swapchainAttachments[0])) {
        testRenderPass->recreate(&swapchainAttachments[0]);
    }
    swapchainFrameBuffers.reserve(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); ++i) {
        drv::drv_assert(testRenderPass->isCompatible(&swapchainAttachments[i]));
        swapchainFrameBuffers.push_back(createResource<drv::Framebuffer>(
          getDevice(), testRenderPass->createFramebuffer(&swapchainAttachments[i])));
    }

    // This only needs recreation, if the renderpass is recreated, but it's good here now for pressure testing
    ShaderObject::DynamicState dynStates = get_dynamic_states(swapchain.getCurrentEXtent());
    shader_global_descriptor::VariantDesc globalDesc;
    shader_test_descriptor::VariantDesc blueVariant;
    shader_test_descriptor::VariantDesc greenVariant;
    shader_test_descriptor::VariantDesc redVariant;
    blueVariant.color = shader_test_descriptor::Color::BLUE;
    greenVariant.color = shader_test_descriptor::Color::GREEN;
    redVariant.color = shader_test_descriptor::Color::RED;
    testShader.prepareGraphicalPipeline(testRenderPass.get(), testSubpass, dynStates, globalDesc,
                                        blueVariant);
    testShader.prepareGraphicalPipeline(testRenderPass.get(), testSubpass, dynStates, globalDesc,
                                        greenVariant);
    testShader.prepareGraphicalPipeline(testRenderPass.get(), testSubpass, dynStates, globalDesc,
                                        redVariant);
}
