#include "game.h"

#include <optional>

#include <util.hpp>

#include <drverror.h>
#include <renderpass.h>

Game::Game(int argc, char* argv[], const EngineConfig& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           const Args& args)
  : Game3D(argc, argv, config, trackingConfig, shaderbinFile, args),
    shaderHeaders(getDevice()),
    shaderObjects(getDevice(), *getShaderBin(), shaderHeaders),
    dynamicStates(drv::DrvShader::DynamicStates::FIXED_SCISSOR,
                  drv::DrvShader::DynamicStates::FIXED_VIEWPORT),
    shaderGlobalDesc(getDevice(), &shaderHeaders.global),
    shaderTestDesc(getDevice(), &shaderHeaders.test),
    testShader(getDevice(), &shaderObjects.test, dynamicStates),
    shaderInputAttachmentDesc(getDevice(), &shaderHeaders.inputatchm),
    inputAttachmentShader(getDevice(), &shaderObjects.inputatchm, dynamicStates) {
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
    colorTagretColorAttachment = testRenderPass->createAttachment(std::move(colorInfo));
    swapchainColorAttachment = testRenderPass->createAttachment(std::move(colorInfo));
    drv::SubpassInfo subpassInfo1;
    subpassInfo1.colorOutputs.push_back(
      {colorTagretColorAttachment, drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL});
    colorSubpass = testRenderPass->createSubpass(std::move(subpassInfo1));
    drv::SubpassInfo subpassInfo2;
    subpassInfo2.colorOutputs.push_back(
      {swapchainColorAttachment, drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL});
    // TODO
    // subpassInfo2.inputs.push_back(
    //   {colorTagretColorAttachment, drv::ImageLayout::SHADER_READ_ONLY_OPTIMAL});
    swapchainSubpass = testRenderPass->createSubpass(std::move(subpassInfo2));
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

void Game::record_cmd_buffer_clear(const RecordData& data, drv::DrvCmdBufferRecorder* recorder) {
    RUNTIME_STAT_SCOPE(gameRecord0);

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
    recorder->cmdImageBarrier({data.renderTarget, drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE,
                               drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false,
                               drv::get_queue_family(data.device, data.renderQueue)});
}

void Game::record_cmd_buffer_render(const RecordData& data, drv::DrvCmdBufferRecorder* recorder) {
    RUNTIME_STAT_SCOPE(gameRecord1);

    drv::ClearValue clearValues[2];
    clearValues[data.swapchainColorAttachment].type = drv::ClearValue::COLOR;
    clearValues[data.colorTagretColorAttachment].type = drv::ClearValue::COLOR;
    clearValues[data.swapchainColorAttachment].value.color =
      drv::ClearColorValue(0.2f, 0.2f, 0.2f, 1.f);
    if (data.variant == 0)
        clearValues[data.colorTagretColorAttachment].value.color =
          drv::ClearColorValue(0.1f, 0.8f, 0.1f, 1.f);
    else
        clearValues[data.colorTagretColorAttachment].value.color =
          drv::ClearColorValue(0.8f, 0.1f, 0.1f, 1.f);
    // clearValues[data.testColorAttachment].value.color = drv::ClearColorValue(255, 255, 255, 255);
    drv::Rect2D renderArea;
    renderArea.extent = data.extent;
    renderArea.offset = {0, 0};
    EngineRenderPass testPass(data.renderPass, recorder, renderArea, data.frameBuffer, clearValues);

    testPass.beginSubpass(data.colorSubpass);
    drv::ClearRect clearRect;
    clearRect.rect.offset = {100, 100};
    clearRect.rect.extent = {data.extent.width - 200, data.extent.height - 200};
    clearRect.baseLayer = 0;
    clearRect.layerCount = 1;
    testPass.clearColorAttachment(data.colorTagretColorAttachment,
                                  drv::ClearColorValue(0.f, 0.7f, 0.7f, 1.f), 1, &clearRect);
    testPass.bindGraphicsShader(get_dynamic_states(data.extent), {}, *data.testShader,
                                data.shaderGlobalDesc, data.shaderTestDesc);
    // testShader.bindGraphicsInfo(ShaderObject::NORMAL_USAGE, testPass,
    //                             get_dynamic_states(swapChainData.extent), &shaderGlobalDesc,
    //                             &data.shaderTestDesc);
    testPass.draw(3, 1, 0, 0);

    testPass.beginSubpass(data.swapchainSubpass);
    testPass.bindGraphicsShader(get_dynamic_states(data.extent), {}, *data.inputShader,
                                data.shaderGlobalDesc, data.shaderInputAttachmentDesc);
    // testShader.bindGraphicsInfo(ShaderObject::NORMAL_USAGE, testPass,
    //                             get_dynamic_states(swapChainData.extent), &shaderGlobalDesc,
    //                             &data.shaderTestDesc);
    testPass.draw(3, 1, 0, 0);

    testPass.end();
}

void Game::record_cmd_buffer_blit(const RecordData& data, drv::DrvCmdBufferRecorder* recorder) {
    RUNTIME_STAT_SCOPE(gameRecord2);

    recorder->cmdImageBarrier({data.renderTarget, drv::IMAGE_USAGE_TRANSFER_SOURCE,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION});

    recorder->cmdImageBarrier({data.targetImage, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION});

    drv::ImageBlit region;
    region.srcSubresource.aspectMask = drv::COLOR_BIT;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount = 1;
    region.srcSubresource.mipLevel = 0;
    region.dstSubresource.aspectMask = drv::COLOR_BIT;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount = 1;
    region.dstSubresource.mipLevel = 0;
    region.srcOffsets[0] = drv::Offset3D{0, 0, 0};
    region.srcOffsets[1] = drv::Offset3D{int(data.extent.width), int(data.extent.height), 1};
    region.dstOffsets[0] = drv::Offset3D{0, 0, 0};
    region.dstOffsets[1] = drv::Offset3D{int(data.extent.width), int(data.extent.height), 1};
    if (region.dstOffsets[1].x > 100)
        region.dstOffsets[1].x = 100;
    if (region.dstOffsets[1].y > 100)
        region.dstOffsets[1].y = 100;
    recorder->cmdBlitImage(data.renderTarget, data.targetImage, 1, &region,
                           drv::ImageFilter::NEAREST);

    // recorder->cmdImageBarrier(
    //   {data.targetImage, drv::IMAGE_USAGE_PRESENT, drv::ImageMemoryBarrier::AUTO_TRANSITION});

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
        testRenderPass->attach(attachments[swapChainData.imageIndex].data());

        RecordData recordData;
        recordData.device = getDevice();
        recordData.targetImage = swapChainData.image;
        recordData.targetView = imageViews[swapChainData.imageIndex].get();
        recordData.renderTarget = renderTarget.get().getImage();
        recordData.renderTargetView = renderTargetView.get();
        recordData.colorTagretColorAttachment = colorTagretColorAttachment;
        recordData.swapchainColorAttachment = swapchainColorAttachment;
        recordData.variant = (frameId / 100) % 2;
        recordData.extent = swapChainData.extent;
        recordData.renderQueue = queues.renderQueue.handle;
        recordData.presentQueue = queues.presentQueue.handle;
        recordData.renderPass = testRenderPass.get();
        recordData.testShader = &testShader;
        recordData.shaderTestDesc = &shaderTestDesc;
        recordData.inputShader = &inputAttachmentShader;
        recordData.shaderInputAttachmentDesc = &shaderInputAttachmentDesc;
        recordData.frameBuffer = swapchainFrameBuffers[swapChainData.imageIndex].get();
        recordData.colorSubpass = colorSubpass;
        recordData.swapchainSubpass = swapchainSubpass;
        recordData.shaderGlobalDesc = &shaderGlobalDesc;

        {
            OneTimeCmdBuffer<RecordData> cmdBuffer(
              "testcmdbuffer_clear", getPhysicalDevice(), getDevice(), queues.renderQueue.handle,
              getCommandBufferBank(), getGarbageSystem(), record_cmd_buffer_clear);
            ExecutionPackage::CommandBufferPackage submission = make_submission_package(
              queues.renderQueue.handle, cmdBuffer.use(std::move(recordData)), getGarbageSystem(),
              ResourceStateValidationMode::IGNORE_FIRST_SUBMISSION, "test_clear");
            submission.waitSemaphores.push_back(
              {swapChainData.imageAvailableSemaphore,
               drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE | drv::IMAGE_USAGE_TRANSFER_DESTINATION});
            testDrawHandle.submit(queues.renderQueue.id, std::move(submission));
        }
        {
            OneTimeCmdBuffer<RecordData> cmdBuffer(
              "testcmdbuffer_render", getPhysicalDevice(), getDevice(), queues.renderQueue.handle,
              getCommandBufferBank(), getGarbageSystem(), record_cmd_buffer_render);
            ExecutionPackage::CommandBufferPackage submission = make_submission_package(
              queues.renderQueue.handle, cmdBuffer.use(std::move(recordData)), getGarbageSystem(),
              ResourceStateValidationMode::IGNORE_FIRST_SUBMISSION, "test_render");
            // submission.waitSemaphores.push_back(
            //   {swapChainData.imageAvailableSemaphore,
            //    drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE | drv::IMAGE_USAGE_TRANSFER_DESTINATION});
            testDrawHandle.submit(queues.renderQueue.id, std::move(submission));
        }
        {
            OneTimeCmdBuffer<RecordData> cmdBuffer(
              "testcmdbuffer_blit", getPhysicalDevice(), getDevice(), queues.renderQueue.handle,
              getCommandBufferBank(), getGarbageSystem(), record_cmd_buffer_blit);
            ExecutionPackage::CommandBufferPackage submission = make_submission_package(
              queues.renderQueue.handle, cmdBuffer.use(std::move(recordData)), getGarbageSystem(),
              ResourceStateValidationMode::IGNORE_FIRST_SUBMISSION, "test_blit");
            submission.signalSemaphores.push_back(swapChainData.renderFinishedSemaphore);
            testDrawHandle.submit(queues.renderQueue.id, std::move(submission));
        }
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
    getGarbageSystem()->useGarbage([this](Garbage* trashBin) {
        testShader.clear(trashBin);
        inputAttachmentShader.clear(trashBin);
    });
    renderTargetView.close();
    renderTarget.close();
    swapchainFrameBuffers.clear();
    attachments.clear();
    imageViews.clear();
}

void Game::createSwapchainResources(const drv::Swapchain& swapchain) {
    imageViews.reserve(swapchain.getImageCount());
    attachments.reserve(swapchain.getImageCount());

    drv::ImageSet::ImageInfo imageInfo;
    imageInfo.imageId = drv::ImageId("targetImage");
    imageInfo.format = drv::ImageFormat::B8G8R8A8_SRGB;
    imageInfo.extent = {swapchain.getCurrentEXtent().width, swapchain.getCurrentEXtent().height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.sampleCount = drv::SampleCount::SAMPLE_COUNT_1;
    // imageInfo.initialLayout = ;
    // imageInfo.familyCount = 0;
    imageInfo.usage = drv::ImageCreateInfo::COLOR_ATTACHMENT_BIT
                      | drv::ImageCreateInfo::INPUT_ATTACHMENT_BIT
                      | drv::ImageCreateInfo::TRANSFER_SRC_BIT;
    imageInfo.type = drv::ImageCreateInfo::TYPE_2D;
    // imageInfo.tiling = ;
    // imageInfo.sharingType = ;
    renderTarget = createResource<drv::ImageSet>(
      getPhysicalDevice(), getDevice(), std::vector<drv::ImageSet::ImageInfo>{imageInfo},
      drv::ImageSet::PreferenceSelector(drv::MemoryType::DEVICE_LOCAL_BIT,
                                        drv::MemoryType::DEVICE_LOCAL_BIT));

    {
        drv::ImageViewCreateInfo createInfo;
        createInfo.image = renderTarget.get().getImage();
        createInfo.type = drv::ImageViewCreateInfo::TYPE_2D;
        createInfo.format = imageInfo.format;
        createInfo.components.r = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.g = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.b = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.components.a = drv::ImageViewCreateInfo::ComponentSwizzle::IDENTITY;
        createInfo.subresourceRange.aspectMask = drv::COLOR_BIT;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.layerCount = 1;
        createInfo.subresourceRange.levelCount = 1;
        renderTargetView = createResource<drv::ImageView>(getDevice(), createInfo);
    }

    drv::RenderPass::AttachmentData colorTargetAttachment;
    colorTargetAttachment.image = renderTarget.get().getImage();
    colorTargetAttachment.view = renderTargetView.get();

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
        attachments.push_back({colorTargetAttachment, swapchainAttachment});
    }
    if (!testRenderPass->isCompatible(attachments[0].data())) {
        testRenderPass->recreate(attachments[0].data());
    }
    swapchainFrameBuffers.reserve(swapchain.getImageCount());
    for (uint32_t i = 0; i < swapchain.getImageCount(); ++i) {
        drv::drv_assert(testRenderPass->isCompatible(attachments[i].data()));
        swapchainFrameBuffers.push_back(createResource<drv::Framebuffer>(
          getDevice(), testRenderPass->createFramebuffer(attachments[i].data())));
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
    testShader.prepareGraphicalPipeline(testRenderPass.get(), swapchainSubpass, dynStates,
                                        globalDesc, blueVariant);
    testShader.prepareGraphicalPipeline(testRenderPass.get(), swapchainSubpass, dynStates,
                                        globalDesc, greenVariant);
    testShader.prepareGraphicalPipeline(testRenderPass.get(), swapchainSubpass, dynStates,
                                        globalDesc, redVariant);
    inputAttachmentShader.prepareGraphicalPipeline(testRenderPass.get(), colorSubpass, dynStates,
                                                   globalDesc, {});
}
