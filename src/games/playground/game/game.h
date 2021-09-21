#pragma once

#include <memory>

#include <drvrenderpass.h>

#include <3dgame.h>
#include <engine.h>
#include <imagestager.h>
#include <serializable.h>

#include <shader_cursor.h>
#include <shader_entityshader.h>
#include <shader_inputatchm.h>
#include <shader_mandelbrot.h>
#include <shader_test.h>
#include <shaderregistry.h>

struct GameOptions final : public IAutoSerializable<GameOptions>
{
    REFLECTABLE((int)mandelBrotLevel, (float)fov)

    GameOptions() : mandelBrotLevel(1), fov(45) {}
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
    ShaderHeaderRegistry shaderHeaders;
    ShaderObjRegistry shaderObjects;
    drv::DrvShader::DynamicStates dynamicStates;
    shader_aglobal_descriptor shaderGlobalDesc;
    shader_threed_descriptor shader3dDescriptor;
    shader_basicshape_descriptor shaderBasicShapeDescriptor;
    shader_forwardshading_descriptor shaderForwardShaderDescriptor;
    shader_entityshader_descriptor entityShaderDesc;
    shader_entityshader entityShader;
    shader_mandelbrot_descriptor mandelbrotDesc;
    shader_mandelbrot mandelbrotShader;
    shader_cursor_descriptor cursorDesc;
    shader_cursor cursorShader;
    shader_fullscreen_descriptor fullscreenDesc;
    shader_sky_descriptor skyDesc;
    shader_sky skyShader;

    std::unique_ptr<drv::RenderPass> renderPass;
    drv::AttachmentId swapchainColorAttachment;
    drv::AttachmentId colorTagretColorAttachment;
    drv::SubpassId backgroundSubpass;
    drv::SubpassId contentSubpass;
    drv::SubpassId foregroundSubpass;
    std::vector<res::ImageView> imageViews;
    std::vector<std::vector<drv::RenderPass::AttachmentData>> attachments;
    std::vector<res::Framebuffer> swapchainFrameBuffers;
    res::ImageSet renderTarget;
    res::ImageView renderTargetView;
    GameOptions gameOptions;

    struct RenderInfo
    {
        const RendererData* rendererData;
        mat4 view;
        mat4 proj;
        mat4 viewProj;
    };

    void recordCmdBufferBackground(const RenderInfo& info, const AcquiredImageData& swapchainData,
                                   EngineCmdBufferRecorder* recorder, EngineRenderPass& pass,
                                   FrameId frameId);
    void recordCmdBufferContent(const RenderInfo& info, const AcquiredImageData& swapchainData,
                                EngineCmdBufferRecorder* recorder, EngineRenderPass& pass,
                                FrameId frameId);
    void recordCmdBufferForeground(const RenderInfo& info, const AcquiredImageData& swapchainData,
                                   EngineCmdBufferRecorder* recorder, EngineRenderPass& pass,
                                   FrameId frameId);
};
