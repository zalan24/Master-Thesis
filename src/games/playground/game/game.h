#pragma once

#include <memory>

#include <drvrenderpass.h>

#include <3dgame.h>
#include <engine.h>
#include <imagestager.h>
#include <serializable.h>

#include <shader_entityshader.h>
#include <shader_inputatchm.h>
#include <shader_mandelbrot.h>
#include <shader_test.h>

struct GameOptions final : public IAutoSerializable<GameOptions>
{
    REFLECTABLE((int)mandelBrotLevel, (float)fov, (float)rotationSpeed, (float)eyeDist,
                (float)eyeHeight)

    GameOptions() : mandelBrotLevel(1), fov(45), rotationSpeed(1), eyeDist(6), eyeHeight(4) {}
};

class Game final : public Game3D
{
 public:
    Game(int argc, char* argv[], const EngineConfig& config,
         const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
         const Resources& resources, const Args& args);
    ~Game() override;

 protected:
    void simulate(FrameId frameId) override;
    void beforeDraw(FrameId frameId) override;
    void record(const AcquiredImageData& swapchainData, EngineCmdBufferRecorder* recorder,
                FrameId frameId) override;
    void lockResources(TemporalResourceLockerDescriptor& resourceDesc, FrameId frameId) override;
    void readback(FrameId frameId) override;
    void releaseSwapchainResources() override;
    void createSwapchainResources(const drv::Swapchain& swapchain) override;

    void recordMenuOptionsUI(FrameId frameId) override;

 private:
    drv::DrvShader::DynamicStates dynamicStates;
    shader_aglobal_descriptor shaderGlobalDesc;
    shader_threed_descriptor shader3dDescriptor;
    shader_basicshape_descriptor shaderBasicShapeDescriptor;
    shader_forwardshading_descriptor shaderForwardShaderDescriptor;
    shader_entityshader_descriptor entityShaderDesc;
    shader_entityshader entityShader;
    shader_test_descriptor shaderTestDesc;
    shader_test testShader;
    shader_mandelbrot_descriptor mandelbrotDesc;
    shader_mandelbrot mandelbrotShader;

    std::unique_ptr<drv::RenderPass> testRenderPass;
    drv::AttachmentId swapchainColorAttachment;
    drv::AttachmentId colorTagretColorAttachment;
    drv::SubpassId colorSubpass;
    drv::SubpassId swapchainSubpass;
    drv::SubpassId imGuiSubpass;
    std::vector<res::ImageView> imageViews;
    std::vector<std::vector<drv::RenderPass::AttachmentData>> attachments;
    std::vector<res::Framebuffer> swapchainFrameBuffers;
    res::ImageSet renderTarget;
    res::ImageView renderTargetView;
    GameOptions gameOptions;

    void recordCmdBufferClear(const AcquiredImageData& swapchainData,
                              EngineCmdBufferRecorder* recorder, FrameId frameId);
    void recordCmdBufferRender(const AcquiredImageData& swapchainData,
                               EngineCmdBufferRecorder* recorder, FrameId frameId);

    //  void recreateViews(uint32_t imageCount, const drv::ImagePtr* images);
    //  void initShader(drv::Extent2D extent);
};
