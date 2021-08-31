#pragma once

#include <memory>

#include <drvrenderpass.h>

#include <3dgame.h>
#include <engine.h>
#include <imagestager.h>
#include <shaderregistry.h>
#include <serializable.h>

#include <shader_inputatchm.h>
#include <shader_mandelbrot.h>
#include <shader_test.h>

struct GameOptions final : public IAutoSerializable<GameOptions>
{
    REFLECTABLE((int) mandelBrotLevel)

    GameOptions() : mandelBrotLevel(1) {}
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
    void record(const AcquiredImageData& swapchainData, drv::DrvCmdBufferRecorder* recorder,
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
    shader_global_descriptor shaderGlobalDesc;
    shader_test_descriptor shaderTestDesc;
    shader_test testShader;
    shader_mandelbrot_descriptor mandelbrotDesc;
    shader_mandelbrot mandelbrotShader;
    shader_inputatchm_descriptor shaderInputAttachmentDesc;
    shader_inputatchm inputAttachmentShader;

    std::unique_ptr<drv::RenderPass> testRenderPass;
    drv::AttachmentId swapchainColorAttachment;
    drv::AttachmentId colorTagretColorAttachment;
    drv::SubpassId colorSubpass;
    drv::SubpassId swapchainSubpass;
    drv::SubpassId imGuiSubpass;
    NodeId transferNode;
    std::vector<res::ImageView> imageViews;
    std::vector<std::vector<drv::RenderPass::AttachmentData>> attachments;
    std::vector<res::Framebuffer> swapchainFrameBuffers;
    res::ImageSet renderTarget;
    res::ImageView renderTargetView;
    GameOptions gameOptions;

    res::ImageSet transferTexture;
    ImageStager testImageStager;

    void recordCmdBufferClear(const AcquiredImageData& swapchainData,
                              drv::DrvCmdBufferRecorder* recorder, FrameId frameId);
    void recordCmdBufferRender(const AcquiredImageData& swapchainData,
                               drv::DrvCmdBufferRecorder* recorder, FrameId frameId);
    void recordCmdBufferBlit(const AcquiredImageData& swapchainData,
                             drv::DrvCmdBufferRecorder* recorder, FrameId frameId);

    //  void recreateViews(uint32_t imageCount, const drv::ImagePtr* images);
    //  void initShader(drv::Extent2D extent);
};
