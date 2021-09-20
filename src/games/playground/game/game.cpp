#include "game.h"

#include <optional>

#include <imgui.h>

#include <util.hpp>

#include <drverror.h>
#include <renderpass.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Game::Game(int argc, char* argv[], const EngineConfig& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           const Resources& _resources, const Args& args)
  : Game3D(argc, argv, config, trackingConfig, shaderbinFile, _resources, args),
    shaderHeaders(getDevice()),
    shaderObjects(getDevice(), *getShaderBin(), shaderHeaders),
    dynamicStates(drv::DrvShader::DynamicStates::FIXED_SCISSOR,
                  drv::DrvShader::DynamicStates::FIXED_VIEWPORT),
    shaderGlobalDesc(getDevice(), &shaderHeaders.aglobal),
    shader3dDescriptor(getDevice(), &shaderHeaders.threed),
    shaderBasicShapeDescriptor(getDevice(), &shaderHeaders.basicshape),
    shaderForwardShaderDescriptor(getDevice(), &shaderHeaders.forwardshading),
    entityShaderDesc(getDevice(), &shaderHeaders.entityshader),
    entityShader(getDevice(), &shaderObjects.entityshader, dynamicStates),
    shaderTestDesc(getDevice(), &shaderHeaders.test),
    testShader(getDevice(), &shaderObjects.test, dynamicStates),
    mandelbrotDesc(getDevice(), &shaderHeaders.mandelbrot),
    mandelbrotShader(getDevice(), &shaderObjects.mandelbrot, dynamicStates) {
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
    drv::SubpassInfo imGuiSubpassInfo;
    imGuiSubpassInfo.colorOutputs.push_back(
      {swapchainColorAttachment, drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL});
    imGuiSubpass = testRenderPass->createSubpass(imGuiSubpassInfo);
    testRenderPass->build();

    initPhysicsEntitySystem();
    initCursorEntitySystem();
    initRenderEntitySystem();

    buildFrameGraph();

    fs::path optionsPath = fs::path{"gameOptions.json"};
    if (fs::exists(optionsPath)) {
        try {
            gameOptions.importFromFile(optionsPath);
        }
        catch (...) {
            gameOptions = {};
        }
    }
}

Game::~Game() {
    fs::path optionsPath = fs::path{"gameOptions.json"};
    if (!fs::exists(optionsPath))
        fs::create_directories(optionsPath.parent_path());
    gameOptions.exportToFile(optionsPath);
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
                                EngineCmdBufferRecorder* recorder, FrameId frameId) {
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
    mandelbrotDesc.setVariant_Quality(
      static_cast<shader_mandelbrot_descriptor::Quality>(gameOptions.mandelBrotLevel));

    recorder->cmdClearImage(swapchainData.image, &clearValue);

    recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE,
                               drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false});
    recorder->cmdImageBarrier({renderTarget.get().getImage(), drv::IMAGE_USAGE_COLOR_OUTPUT_WRITE,
                               drv::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false});
}

void Game::recordCmdBufferRender(const AcquiredImageData& swapchainData,
                                 EngineCmdBufferRecorder* recorder, FrameId frameId) {
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
    EngineRenderPass testPass(testRenderPass.get(), recorder->get(), renderArea,
                              swapchainFrameBuffers[swapchainData.imageIndex].get(), clearValues);

    testPass.beginSubpass(colorSubpass);  // renders to target texture
    drv::ClearRect clearRect;
    clearRect.rect.offset = {100, 100};
    clearRect.rect.extent = {swapchainData.extent.width - 200, swapchainData.extent.height - 200};
    clearRect.baseLayer = 0;
    clearRect.layerCount = 1;
    testPass.clearColorAttachment(colorTagretColorAttachment,
                                  drv::ClearColorValue(0.f, 0.7f, 0.7f, 1.f), 1, &clearRect);

    recorder->bindGraphicsShader(testPass, get_dynamic_states(swapchainData.extent), {}, testShader,
                                 &shaderGlobalDesc, &shaderTestDesc);
    // testShader.bindGraphicsInfo(ShaderObject::NORMAL_USAGE, testPass,
    //                             get_dynamic_states(swapChainextent), &shaderGlobalDesc,
    //                             &shaderTestDesc);
    testPass.draw(3, 1, 0, 0);

    testPass.beginSubpass(swapchainSubpass);  // renders to swapchain
    // recorder->bindGraphicsShader(testPass, get_dynamic_states(swapchainData.extent), {}, inputAttachmentShader,
    //                             &shaderGlobalDesc, &shaderInputAttachmentDesc);
    // // testShader.bindGraphicsInfo(ShaderObject::NORMAL_USAGE, testPass,
    // //                             get_dynamic_states(swapChainextent), &shaderGlobalDesc,
    // //                             &shaderTestDesc);
    // testPass.draw(3, 1, 0, 0);
    mandelbrotDesc.set_exitColor(vec3(0, 0, 0));
    mandelbrotDesc.set_midColor(
      lerp(vec3(1, 0, 0), vec3(0, 1, 0), float(sin(double(frameId) * 0.01) * 0.5 + 0.5)));
    mandelbrotDesc.set_peakColor(vec3(1, 1, 1));
    recorder->bindGraphicsShader(testPass, get_dynamic_states(swapchainData.extent), {},
                                 mandelbrotShader, &mandelbrotDesc);
    testPass.draw(6, 1, 0, 0);

    uint32_t planeResolution = 2;
    uint32_t boxResolution = 2;
    uint32_t sphereResolution = 20;
    float brightness = 0.5;
    const RendererData rendererData = getRenderData(frameId);
    shader3dDescriptor.set_eyePos(rendererData.eyePos);
    mat4 view =
      glm::lookAtLH(rendererData.eyePos, rendererData.eyePos + rendererData.eyeDir, vec3(0, 1, 0));
    mat4 proj =
      glm::perspective(glm::radians(gameOptions.fov * 2), rendererData.ratio, 0.01f, 150.0f);
    mat4 bsToGoodTm(1.f);
    bsToGoodTm[1] = -bsToGoodTm[1];
    bsToGoodTm[2] = -bsToGoodTm[2];
    proj = proj * bsToGoodTm;
    shader3dDescriptor.set_viewProj(proj * view);
    shaderForwardShaderDescriptor.set_ambientLight(vec3(0.1, 0.1, 0.1) * brightness);
    shaderForwardShaderDescriptor.set_sunDir(glm::normalize(vec3(-0.2, -0.8, 0.4)));
    shaderForwardShaderDescriptor.set_sunLight(vec3(1.0, 0.8, 0.7) * brightness);
    shaderForwardShaderDescriptor.setVariant_renderPass(
      shader_forwardshading_descriptor::Renderpass::COLOR_PASS);

    uint32_t numEntities = getNumEntitiesToRender();
    for (uint32_t i = 0; i < numEntities; ++i) {
        const EntityRenderData& data = getEntitiesToRender()[i];
        shader3dDescriptor.set_modelTm(data.modelTm);
        entityShaderDesc.set_entityAlbedo(data.albedo);

        if (data.shape == "plane") {
            shaderBasicShapeDescriptor.set_resolution(planeResolution);
            shaderBasicShapeDescriptor.setVariant_Shape(
              shader_basicshape_descriptor::Shape::SHAPE_PLANE);
            recorder->bindGraphicsShader(
              testPass, get_dynamic_states(swapchainData.extent), {}, entityShader,
              &shader3dDescriptor, &shaderForwardShaderDescriptor, &shaderBasicShapeDescriptor,
              &shaderGlobalDesc, &entityShaderDesc);
            testPass.draw(6 * planeResolution * planeResolution, 1, 0, 0);
        }
        else if (data.shape == "box") {
            shaderBasicShapeDescriptor.set_resolution(boxResolution);
            shaderBasicShapeDescriptor.setVariant_Shape(
              shader_basicshape_descriptor::Shape::SHAPE_BOX);
            recorder->bindGraphicsShader(
              testPass, get_dynamic_states(swapchainData.extent), {}, entityShader,
              &shader3dDescriptor, &shaderForwardShaderDescriptor, &shaderBasicShapeDescriptor,
              &shaderGlobalDesc, &entityShaderDesc);
            testPass.draw(6 * 6 * boxResolution * boxResolution, 1, 0, 0);
        }
        else if (data.shape == "sphere") {
            shaderBasicShapeDescriptor.set_resolution(sphereResolution);
            shaderBasicShapeDescriptor.setVariant_Shape(
              shader_basicshape_descriptor::Shape::SHAPE_SPHERE);
            recorder->bindGraphicsShader(
              testPass, get_dynamic_states(swapchainData.extent), {}, entityShader,
              &shader3dDescriptor, &shaderForwardShaderDescriptor, &shaderBasicShapeDescriptor,
              &shaderGlobalDesc, &entityShaderDesc);
            testPass.draw(6 * sphereResolution * sphereResolution, 1, 0, 0);
        }
        else
            throw std::runtime_error("Unknown shape: " + data.shape);
    }

    testPass.beginSubpass(imGuiSubpass);
    recordImGui(swapchainData, recorder, frameId);

    testPass.end();
}

void Game::lockResources(TemporalResourceLockerDescriptor&, FrameId) {
}

void Game::record(const AcquiredImageData& swapchainData, EngineCmdBufferRecorder* recorder,
                  FrameId frameId) {
    testRenderPass->attach(attachments[swapchainData.imageIndex].data());

    recordCmdBufferClear(swapchainData, recorder, frameId);
    recordCmdBufferRender(swapchainData, recorder, frameId);
    recorder->cmdImageBarrier({swapchainData.image, drv::IMAGE_USAGE_PRESENT,
                               drv::ImageMemoryBarrier::AUTO_TRANSITION, false});
}

void Game::simulate(FrameId frameId) {
    UNUSED(frameId);
    // std::cout << "Simulate: " << frameId << std::endl;
}

void Game::beforeDraw(FrameId) {
}

void Game::readback(FrameId) {
}

void Game::releaseSwapchainResources() {
    getGarbageSystem()->useGarbage([this](Garbage* trashBin) {
        testShader.clear(trashBin);
        mandelbrotShader.clear(trashBin);
        entityShader.clear(trashBin);
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
    shader_aglobal_descriptor::VariantDesc globalDesc;
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

    initImGui(testRenderPass.get());
}

void Game::recordMenuOptionsUI(FrameId) {
    if (ImGui::BeginMenu("Camera")) {
        ImGui::DragFloat("Fov", &gameOptions.fov, 0.1f, 10, 80, "%.1f deg");
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Mandelbrot level")) {
        int from = static_cast<int>(shader_mandelbrot_descriptor::Quality::QUALITY1);
        int to = static_cast<int>(shader_mandelbrot_descriptor::Quality::QUALITY10);
        char label[128];
        for (int i = from; i <= to; ++i) {
            sprintf(label, "Quality %d", i + 1);
            ImGui::RadioButton(label, &gameOptions.mandelBrotLevel, i);
        }
        ImGui::EndMenu();
    }
}
