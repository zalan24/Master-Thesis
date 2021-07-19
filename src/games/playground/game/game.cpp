#include "game.h"

#include <optional>

#include <util.hpp>

#include <drverror.h>
#include <renderpass.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <CImg.h>
#include <stb_image_write.h>

Game::Game(int argc, char* argv[], const EngineConfig& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           const Resources& _resources, const Args& args)
  : Game3D(argc, argv, config, trackingConfig, shaderbinFile, _resources, args),
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

    transferNode = getFrameGraph().addNode(
      FrameGraph::Node("transfer", FrameGraph::BEFORE_DRAW_STAGE | FrameGraph::READBACK_STAGE));

    getFrameGraph().addDependency(getMainRecordNode(), FrameGraph::CpuDependency{transferNode, FrameGraph::BEFORE_DRAW_STAGE, FrameGraph::RECORD_STAGE, 0});
    getFrameGraph().addDependency(transferNode, FrameGraph::GpuCpuDependency{getMainRecordNode(), FrameGraph::READBACK_STAGE, 0});


    initPhysicsEntitySystem();
    initRenderEntitySystem();

    // TODO present node could be inside of record end node???
    buildFrameGraph();
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

void Game::recordCmdBufferClear(const AcquiredImageData& swapchainData,
                                drv::DrvCmdBufferRecorder* recorder, FrameId frameId) {
    recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, true});
    drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
    if ((frameId / 100) % 2 == 0) {
        clearValue = drv::ClearColorValue(1.f, 1.f, 0.f, 1.f);
        shaderTestDesc.setVariant_Color(shader_test_descriptor::Color::BLUE);
    }
    else {
        clearValue = drv::ClearColorValue(0.f, 1.f, 1.f, 1.f);
        shaderTestDesc.setVariant_Color(shader_test_descriptor::Color::RED);
    }

    recorder->cmdClearImage(swapchainData.image, &clearValue);

    recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE,
                               drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false});
    recorder->cmdImageBarrier({renderTarget.get().getImage(), drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE,
                               drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false});
}

void Game::recordCmdBufferRender(const AcquiredImageData& swapchainData,
                                 drv::DrvCmdBufferRecorder* recorder, FrameId frameId) {
    recorder->cmdImageBarrier({transferTexture.get().getImage(0),
                               drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION});

    // testImageStager->transferFromStager(recorder, stagerId);

    drv::ClearValue clearValues[2];
    clearValues[swapchainColorAttachment].type = drv::ClearValue::COLOR;
    clearValues[colorTagretColorAttachment].type = drv::ClearValue::COLOR;
    clearValues[swapchainColorAttachment].value.color = drv::ClearColorValue(0.2f, 0.2f, 0.2f, 1.f);
    if ((frameId / 100) % 2 == 0)
        clearValues[colorTagretColorAttachment].value.color =
          drv::ClearColorValue(0.1f, 0.8f, 0.1f, 1.f);
    else
        clearValues[colorTagretColorAttachment].value.color =
          drv::ClearColorValue(0.8f, 0.1f, 0.1f, 1.f);
    // clearValues[testColorAttachment].value.color = drv::ClearColorValue(255, 255, 255, 255);
    drv::Rect2D renderArea;
    renderArea.extent = swapchainData.extent;
    renderArea.offset = {0, 0};
    EngineRenderPass testPass(testRenderPass.get(), recorder, renderArea,
                              swapchainFrameBuffers[swapchainData.imageIndex].get(), clearValues);

    testPass.beginSubpass(colorSubpass);
    drv::ClearRect clearRect;
    clearRect.rect.offset = {100, 100};
    clearRect.rect.extent = {swapchainData.extent.width - 200, swapchainData.extent.height - 200};
    clearRect.baseLayer = 0;
    clearRect.layerCount = 1;
    testPass.clearColorAttachment(colorTagretColorAttachment,
                                  drv::ClearColorValue(0.f, 0.7f, 0.7f, 1.f), 1, &clearRect);
    testPass.bindGraphicsShader(get_dynamic_states(swapchainData.extent), {}, testShader,
                                &shaderGlobalDesc, &shaderTestDesc);
    // testShader.bindGraphicsInfo(ShaderObject::NORMAL_USAGE, testPass,
    //                             get_dynamic_states(swapChainextent), &shaderGlobalDesc,
    //                             &shaderTestDesc);
    testPass.draw(3, 1, 0, 0);

    testPass.beginSubpass(swapchainSubpass);
    testPass.bindGraphicsShader(get_dynamic_states(swapchainData.extent), {}, inputAttachmentShader,
                                &shaderGlobalDesc, &shaderInputAttachmentDesc);
    // testShader.bindGraphicsInfo(ShaderObject::NORMAL_USAGE, testPass,
    //                             get_dynamic_states(swapChainextent), &shaderGlobalDesc,
    //                             &shaderTestDesc);
    testPass.draw(3, 1, 0, 0);

    testPass.end();

    drv::TextureInfo texInfo = drv::get_texture_info(transferTexture.get().getImage(0));
    if (swapchainData.extent.height >= 100 + texInfo.extent.height
        && swapchainData.extent.width >= 100 + texInfo.extent.width) {
        recorder->cmdImageBarrier({transferTexture.get().getImage(0),
                                   drv::IMAGE_USAGE_TRANSFER_SOURCE,
                                   drv::ImageMemoryBarrier::AUTO_TRANSITION});

        recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
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
        region.srcOffsets[1] =
          drv::Offset3D{int(texInfo.extent.width), int(texInfo.extent.height), 1};
        region.dstOffsets[0] = drv::Offset3D{100, 0, 0};
        region.dstOffsets[1] =
          drv::Offset3D{int(texInfo.extent.width) + 100, int(texInfo.extent.height), 1};
        recorder->cmdBlitImage(transferTexture.get().getImage(0), swapchainData.image, 1, &region,
                               drv::ImageFilter::NEAREST);
    }

    // recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
    //                            drv::ImageMemoryBarrier::AUTO_TRANSITION, true,
    //                            drv::get_queue_family(device, renderQueue)});
    // drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
    // recorder->cmdClearImage(swapchainData.image, &clearValue);
}

void Game::recordCmdBufferBlit(const AcquiredImageData& swapchainData,
                               drv::DrvCmdBufferRecorder* recorder, FrameId) {
    recorder->cmdImageBarrier({renderTarget.get().getImage(), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION});

    recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_TRANSFER_DESTINATION,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION});

    recorder->cmdImageBarrier({transferTexture.get().getImage(0),
                               drv::IMAGE_USAGE_TRANSFER_DESTINATION,
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
    region.srcOffsets[1] =
      drv::Offset3D{int(swapchainData.extent.width), int(swapchainData.extent.height), 1};
    region.dstOffsets[0] = drv::Offset3D{0, 0, 0};
    region.dstOffsets[1] =
      drv::Offset3D{int(swapchainData.extent.width), int(swapchainData.extent.height), 1};
    if (region.dstOffsets[1].x > 100)
        region.dstOffsets[1].x = 100;
    if (region.dstOffsets[1].y > 100)
        region.dstOffsets[1].y = 100;
    recorder->cmdBlitImage(renderTarget.get().getImage(), swapchainData.image, 1, &region,
                           drv::ImageFilter::NEAREST);

    drv::TextureInfo texInfo = drv::get_texture_info(transferTexture.get().getImage(0));
    region.dstOffsets[0] = drv::Offset3D{0, 0, 0};
    region.dstOffsets[1] = drv::Offset3D{int(texInfo.extent.width), int(texInfo.extent.height), 1};
    recorder->cmdBlitImage(renderTarget.get().getImage(), transferTexture.get().getImage(0), 1,
                           &region, drv::ImageFilter::LINEAR);

    // recorder->cmdImageBarrier({transferTexture.get().getImage(0), drv::IMAGE_USAGE_TRANSFER_SOURCE,
    //                            drv::ImageMemoryBarrier::AUTO_TRANSITION});
    // testImageStager->transferToStager(recorder, stagerId);

    // recorder->cmdImageBarrier(
    //   {swapchainData.image, drv::IMAGE_USAGE_PRESENT, drv::ImageMemoryBarrier::AUTO_TRANSITION});

    // drv::ClearColorValue clearValue(1.f, 1.f, 0.f, 1.f);
    // recorder->cmdClearImage(swapChainimage, &clearValue);
    // /// --- clear ---
    // TODO according to vulkan spec https://khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueuePresentKHR.html
    // memory is made visible to all read operations (add this to tracker?) -- only available memory
}

void Game::lockResources(TemporalResourceLockerDescriptor& resourceDesc, FrameId frameId) {
    ImageStager::StagerId stagerId = testImageStager.getStagerId(frameId);
    testImageStager.lockResource(resourceDesc, ImageStager::UPLOAD, stagerId);
}

void Game::record(const AcquiredImageData& swapchainData, drv::DrvCmdBufferRecorder* recorder,
                  FrameId frameId) {
    ImageStager::StagerId stagerId = testImageStager.getStagerId(frameId);
    // std::cout << "Record: " << frameId << std::endl;
    // LOG_F(INFO, "Frame %lld  Swapchain image: %d Image: %p", frameId, swapchainData.imageIndex,
    //       static_cast<const void*>(swapchainData.image));

    // drv::RenderPass::AttachmentData testImageInfo[1];
    // testImageInfo[testColorAttachment].image = swapchainData.image;
    // testImageInfo[testColorAttachment].view = imageViews[swapchainData.imageIndex].get();
    // if (testRenderPass->needRecreation(testImageInfo)) {
    //     for (auto& framebuffer : frameBuffers)
    //         framebuffer = {};
    //     testRenderPass->recreate(testImageInfo);
    //     initShader(swapchainData.extent);
    // }
    // if (!frameBuffers[swapchainData.imageIndex]
    //     || !frameBuffers[swapchainData.imageIndex].get())
    //     frameBuffers[swapchainData.imageIndex] = createResource<drv::Framebuffer>(
    //       getDevice(), testRenderPass->createFramebuffer(testImageInfo));
    testRenderPass->attach(attachments[swapchainData.imageIndex].data());

    recordCmdBufferClear(swapchainData, recorder, frameId);
    testImageStager.transferFromStager(recorder, stagerId);
    recordCmdBufferRender(swapchainData, recorder, frameId);
    recordCmdBufferBlit(swapchainData, recorder, frameId);
    drawEntities(recorder);
    recorder->cmdImageBarrier({transferTexture.get().getImage(0), drv::IMAGE_USAGE_TRANSFER_SOURCE,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION});
    testImageStager.transferToStager(recorder, stagerId);
    recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_PRESENT,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, false});
}

void Game::simulate(FrameId frameId) {
    UNUSED(frameId);
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    // std::cout << "Simulate: " << frameId << std::endl;
}

void Game::beforeDraw(FrameId frameId) {
    TemporalResourceLockerDescriptor resourceDesc;
    ImageStager::StagerId stagerId = testImageStager.getStagerId(frameId);
    testImageStager.lockResource(resourceDesc, ImageStager::UPLOAD, stagerId);
    if (FrameGraph::NodeHandle testDrawHandle = getFrameGraph().acquireNode(
          transferNode, FrameGraph::BEFORE_DRAW_STAGE, frameId, resourceDesc);
        testDrawHandle) {
        drv::DeviceSize size;
        drv::DeviceSize rowPitch;
        drv::DeviceSize arrayPitch;
        drv::DeviceSize depthPitch;
        testImageStager.getMemoryData(stagerId, 0, 0, size, rowPitch, arrayPitch, depthPitch);
        drv::TextureInfo texInfo = drv::get_texture_info(transferTexture.get().getImage(0));
        cimg_library::CImg<unsigned char> textImage(texInfo.extent.width, texInfo.extent.height, 1,
                                                    3, 255);
        unsigned char black[] = {0, 0, 0};
        unsigned char white[] = {255, 255, 255};

        // Draw black text on cyan
        textImage.draw_text(8, 100, "Stager: %.2d, Frame: %.4d", black, white, 1, 16, stagerId,
                            frameId);
        StackMemory::MemoryHandle<uint32_t> pixels(size / 4, TEMPMEM);
        for (uint32_t y = 0; y < texInfo.extent.height; ++y) {
            for (uint32_t x = 0; x < texInfo.extent.width; ++x) {
                uint32_t r, g, b, a;
                r = textImage.data()[x + texInfo.extent.width * y
                                     + texInfo.extent.width * texInfo.extent.height * 0];
                g = textImage.data()[x + texInfo.extent.width * y
                                     + texInfo.extent.width * texInfo.extent.height * 1];
                b = textImage.data()[x + texInfo.extent.width * y
                                     + texInfo.extent.width * texInfo.extent.height * 2];
                a = 255;
                pixels[StackMemory::size_t(x + y * rowPitch / 4)] =
                  (a << 24) + (b << 16) + (g << 8) + r;
            }
        }
        if (frameId == 0) {
            stbi_write_png("test_image_out_generated.png", int(texInfo.extent.width),
                           int(texInfo.extent.height), 4, pixels, int(rowPitch));
            textImage.save_bmp("test_image_out_gen_cimg.bmp");
        }
        // layer*arrayPitch + z*depthPitch + y*rowPitch + x*elementSize + offset
        testImageStager.setData(pixels, 0, 0, stagerId, testDrawHandle.getLock());
    }
}

void Game::readback(FrameId frameId) {
    TemporalResourceLockerDescriptor resourceDesc;
    ImageStager::StagerId stagerId = testImageStager.getStagerId(frameId);
    testImageStager.lockResource(resourceDesc, ImageStager::DOWNLOAD, stagerId);
    if (FrameGraph::NodeHandle testDrawHandle = getFrameGraph().acquireNode(
          transferNode, FrameGraph::READBACK_STAGE, frameId, resourceDesc);
        testDrawHandle) {
        drv::DeviceSize size;
        drv::DeviceSize rowPitch;
        drv::DeviceSize arrayPitch;
        drv::DeviceSize depthPitch;
        testImageStager.getMemoryData(stagerId, 0, 0, size, rowPitch, arrayPitch, depthPitch);
        StackMemory::MemoryHandle<uint32_t> pixels(size / 4, TEMPMEM);
        testImageStager.getData(pixels, 0, 0, stagerId, testDrawHandle.getLock());
        if (frameId == 19) {
            drv::TextureInfo texInfo = drv::get_texture_info(transferTexture.get().getImage(0));
            stbi_write_png("test_image_out.png", int(texInfo.extent.width),
                           int(texInfo.extent.height), 4, pixels, int(rowPitch));
        }
    }
}

void Game::releaseSwapchainResources() {
    getGarbageSystem()->useGarbage([this](Garbage* trashBin) {
        testShader.clear(trashBin);
        inputAttachmentShader.clear(trashBin);
    });
    renderTargetView.close();
    renderTarget.close();
    testImageStager.clear();
    transferTexture.close();
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

    drv::ImageSet::ImageInfo transferImageInfo;
    transferImageInfo.imageId = drv::ImageId("transferTex");
    transferImageInfo.format = drv::ImageFormat::R8G8B8A8_UNORM;
    transferImageInfo.extent = {256, 256, 1};
    transferImageInfo.mipLevels = 1;
    transferImageInfo.arrayLayers = 1;
    transferImageInfo.sampleCount = drv::SampleCount::SAMPLE_COUNT_1;
    // transferImageInfo.initialLayout = ;
    // transferImageInfo.familyCount = 0;
    transferImageInfo.usage =
      drv::ImageCreateInfo::TRANSFER_DST_BIT | drv::ImageCreateInfo::TRANSFER_SRC_BIT;
    transferImageInfo.type = drv::ImageCreateInfo::TYPE_2D;
    // transferImageInfo.tiling = ;
    // transferImageInfo.sharingType = ;
    transferTexture = createResource<drv::ImageSet>(
      getPhysicalDevice(), getDevice(), std::vector<drv::ImageSet::ImageInfo>{transferImageInfo},
      drv::ImageSet::PreferenceSelector(drv::MemoryType::DEVICE_LOCAL_BIT,
                                        drv::MemoryType::DEVICE_LOCAL_BIT));
    testImageStager = ImageStager(this, transferTexture.get().getImage(0), getMaxFramesInFlight(),
                                  ImageStager::BOTH);

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
