#pragma once

#include <drvcmdbuffer.h>
#include <drvresourcelocker.h>
#include <drvresourceptrs.hpp>

#include <drv_wrappers.h>

#include "resources.hpp"

class Engine;

class BufferStager
{
 public:
    using StagerId = uint32_t;

    enum Usage
    {
        DOWNLOAD,
        UPLOAD,
        BOTH
    };

    BufferStager() = default;
    BufferStager(Engine* engine, drv::BufferPtr buffer, const drv::BufferSubresourceRange& subres,
                 uint32_t numStagers, Usage usage);
    BufferStager(Engine* engine, drv::BufferPtr buffer, uint32_t numStagers, Usage usage);

    BufferStager(const BufferStager&) = delete;
    BufferStager& operator=(const BufferStager&) = delete;
    BufferStager(BufferStager&& other);
    BufferStager& operator=(BufferStager&& other);

    ~BufferStager();

    void transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager);
    void transferFromStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                            const drv::BufferSubresourceRange& subres);
    void transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager);
    void transferToStager(drv::DrvCmdBufferRecorder* recorder, StagerId stager,
                          const drv::BufferSubresourceRange& subres);

    void clear();

    void setData(const void* srcData, const drv::BufferSubresourceRange& range, StagerId stager,
                 const drv::ResourceLocker::Lock& lock);
    void setData(const void* srcData, StagerId stager, const drv::ResourceLocker::Lock& lock);
    void getData(void* dstData, const drv::BufferSubresourceRange& subres, StagerId stager,
                 const drv::ResourceLocker::Lock& lock);
    void getData(void* dstData, StagerId stager, const drv::ResourceLocker::Lock& lock);

    StagerId getStagerId(FrameId frame) const;
    void lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage,
                      StagerId stager) const;
    void lockResource(drv::ResourceLockerDescriptor& descriptor, Usage usage, StagerId stager,
                      drv::BufferSubresourceRange subres) const;

    operator bool() const { return !drv::is_null_ptr(buffer); }

    uint32_t getNumStagers() const { return numStagers; }

    drv::BufferPtr getBuffer() const { return buffer; }

 private:
    drv::LogicalDevicePtr device = drv::get_null_ptr<drv::LogicalDevicePtr>();
    drv::BufferPtr buffer = drv::get_null_ptr<drv::BufferPtr>();
    res::BufferSet stagers;
    drv::BufferSubresourceRange subresource;
    uint32_t numStagers = 0;

    bool checkSubres(const drv::BufferSubresourceRange& subres) const;

    void close();
};
