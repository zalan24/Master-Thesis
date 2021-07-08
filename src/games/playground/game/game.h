#pragma once

#include <memory>

#include <drvrenderpass.h>

#include <3dgame.h>
#include <engine.h>
#include <imagestager.h>
#include <shaderregistry.h>

#include <shader_inputatchm.h>
#include <shader_test.h>

class Game final : public Game3D
{
 public:
    Game(int argc, char* argv[], const EngineConfig& config,
         const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
         const Args& args);
    ~Game() override;

 protected:
    void simulate(FrameId frameId) override;
    void beforeDraw(FrameId frameId) override;
    AcquiredImageData record(FrameId frameId) override;
    void readback(FrameId frameId) override;
    void releaseSwapchainResources() override;
    void createSwapchainResources(const drv::Swapchain& swapchain) override;

 private:
    ShaderHeaderRegistry shaderHeaders;
    ShaderObjRegistry shaderObjects;
    drv::DrvShader::DynamicStates dynamicStates;
    shader_global_descriptor shaderGlobalDesc;
    shader_test_descriptor shaderTestDesc;
    shader_test testShader;
    shader_inputatchm_descriptor shaderInputAttachmentDesc;
    shader_inputatchm inputAttachmentShader;

    std::unique_ptr<drv::RenderPass> testRenderPass;
    drv::AttachmentId swapchainColorAttachment;
    drv::AttachmentId colorTagretColorAttachment;
    drv::SubpassId colorSubpass;
    drv::SubpassId swapchainSubpass;
    FrameGraph::NodeId testDraw;
    std::vector<res::ImageView> imageViews;
    std::vector<std::vector<drv::RenderPass::AttachmentData>> attachments;
    std::vector<res::Framebuffer> swapchainFrameBuffers;
    res::ImageSet renderTarget;
    res::ImageView renderTargetView;

    res::ImageSet transferTexture;
    ImageStager testImageStager;

    // TODO automatize this kind of thing + check resource states
    struct RecordData
    {
        drv::LogicalDevicePtr device;
        drv::ImagePtr targetImage;
        drv::ImageViewPtr targetView;
        drv::ImagePtr renderTarget;
        drv::ImagePtr transferImage;
        drv::ImageViewPtr renderTargetView;
        drv::AttachmentId swapchainColorAttachment;
        drv::AttachmentId colorTagretColorAttachment;
        uint32_t variant;
        drv::Extent2D extent;
        drv::QueuePtr renderQueue;
        drv::QueuePtr presentQueue;
        drv::FramebufferPtr frameBuffer;
        drv::RenderPass* renderPass;
        drv::SubpassId colorSubpass;
        drv::SubpassId swapchainSubpass;
        shader_test* testShader;
        shader_test_descriptor* shaderTestDesc;
        shader_global_descriptor* shaderGlobalDesc;
        shader_inputatchm* inputShader;
        shader_inputatchm_descriptor* shaderInputAttachmentDesc;
        ImageStager::StagerId stagerId;
        ImageStager *testImageStager;
        bool doBlit;
        bool operator==(const RecordData& rhs) const {
            return device == rhs.device && targetImage == rhs.targetImage
                   && targetView == rhs.targetView && renderTarget == rhs.renderTarget
                   && renderTargetView == rhs.renderTargetView
                   && swapchainColorAttachment == rhs.swapchainColorAttachment
                   && colorTagretColorAttachment == rhs.colorTagretColorAttachment
                   && variant == rhs.variant && extent == rhs.extent
                   && renderQueue == rhs.renderQueue && presentQueue == rhs.presentQueue
                   && frameBuffer == rhs.frameBuffer && renderPass == rhs.renderPass
                   && colorSubpass == rhs.colorSubpass && swapchainSubpass == rhs.swapchainSubpass
                   && testShader == rhs.testShader && shaderTestDesc == rhs.shaderTestDesc
                   && shaderGlobalDesc == rhs.shaderGlobalDesc && inputShader == rhs.inputShader
                   && shaderInputAttachmentDesc == rhs.shaderInputAttachmentDesc
                   && doBlit == rhs.doBlit && stagerId == rhs.stagerId && transferImage == rhs.transferImage && testImageStager == rhs.testImageStager;
        }
        bool operator!=(const RecordData& rhs) const { return !(*this == rhs); }
    };

    static void record_cmd_buffer_clear(const RecordData& data,
                                        drv::DrvCmdBufferRecorder* recorder);
    static void record_cmd_buffer_render(const RecordData& data,
                                         drv::DrvCmdBufferRecorder* recorder);
    static void record_cmd_buffer_blit(const RecordData& data, drv::DrvCmdBufferRecorder* recorder);

    //  void recreateViews(uint32_t imageCount, const drv::ImagePtr* images);
    //  void initShader(drv::Extent2D extent);
};
