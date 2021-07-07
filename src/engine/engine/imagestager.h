#pragma once

#include <drvcmdbuffer.h>
#include <drvresourcelocker.h>
#include <drvresourceptrs.hpp>

#include <drv_wrappers.h>

#include "resources.hpp"

class Engine;

class ImageStager
{
 public:
    using StagerId = uint32_t;

    enum Usage
    {
        DOWNLOAD,
        UPLOAD,
        BOTH
    };

    ImageStager() = default;
    ImageStager(Engine* engine, drv::ImagePtr image, drv::ImageFormat format,
                const drv::ImageSubresourceRange& subres, uint32_t numStagers, Usage usage);
    ImageStager(Engine* engine, drv::ImagePtr image, drv::ImageFormat format, uint32_t numStagers,
                Usage usage);

    ImageStager(const ImageStager&) = delete;
    ImageStager& operator=(const ImageStager&) = delete;
    ImageStager(ImageStager&& other);
    ImageStager& operator=(ImageStager&& other);

    ~ImageStager();

    void transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager);
    void transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager, uint32_t layer,
                            uint32_t mip);
    void transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                            const drv::ImageSubresourceRange& subres);
    void transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager);
    void transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager, uint32_t layer,
                          uint32_t mip);
    void transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                          const drv::ImageSubresourceRange& subres);

    void clear();

    void getMemoryData(StagerId stager, uint32_t layer, uint32_t mip, drv::DeviceSize& size,
                       drv::DeviceSize& rowPitch, drv::DeviceSize& arrayPitch,
                       drv::DeviceSize& depthPitch) const;
    void setData(const void* srcData, uint32_t layer, uint32_t mip, StagerId stager,
                 const drv::ResourceLocker::Lock& lock);
    void getData(void* dstData, uint32_t layer, uint32_t mip, StagerId stager,
                 const drv::ResourceLocker::Lock& lock);

    StagerId getStagerId(FrameId frame) const;
    void lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                      StagerId stager) const;
    void lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage, StagerId stager,
                      uint32_t layer, uint32_t mip) const;
    void lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage, StagerId stager,
                      drv::ImageSubresourceRange subres) const;

    operator bool() const { return !drv::is_null_ptr(image); }

 private:
    drv::LogicalDevicePtr device;
    drv::ImagePtr image = drv::get_null_ptr<drv::ImagePtr>();
    res::ImageSet stagers;
    drv::ImageSubresourceRange subresource;
    uint32_t numStagers = 0;
    uint32_t mipOffset = 0;
    uint32_t layerOffset = 0;
    uint32_t mipCount = 0;
    uint32_t layerCount = 0;

    bool checkSubres(const drv::ImageSubresourceRange& subres, uint32_t& subresMipCount,
                     uint32_t& subresLayerCount) const;

    void close();
};
