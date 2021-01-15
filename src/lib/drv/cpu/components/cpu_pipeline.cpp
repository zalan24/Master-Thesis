
#include "cpu_pipeline.h"

#include <drverror.h>

drv::PipelineLayoutPtr drv_cpu::create_pipeline_layout(drv::LogicalDevicePtr,
                                                       const drv::PipelineLayoutCreateInfo* info) {
    CpuPipelineLayout* layout = new CpuPipelineLayout;
    try {
        return reinterpret_cast<drv::PipelineLayoutPtr>(layout);
    }
    catch (...) {
        delete layout;
        throw;
    }
}

bool drv_cpu::destroy_pipeline_layout(drv::LogicalDevicePtr, drv::PipelineLayoutPtr layout) {
    delete reinterpret_cast<CpuPipelineLayout*>(layout);
    return true;
}

bool drv_cpu::create_compute_pipeline(drv::LogicalDevicePtr, unsigned int count,
                                      const drv::ComputePipelineCreateInfo* infos,
                                      drv::ComputePipelinePtr* _pipelines) {
    CpuComputePipeline** pipelines = reinterpret_cast<CpuComputePipeline**>(_pipelines);
    unsigned int c = 0;
    try {
        for (unsigned int i = 0; i < count; ++i) {
            pipelines[i] = new CpuComputePipeline;
            c++;
            pipelines[i]->computeModule = infos[i].stage.module;
        }
        return true;
    }
    catch (...) {
        for (unsigned int i = 0; i < c; ++i)
            delete pipelines[i];
        throw;
    }
}

bool drv_cpu::destroy_compute_pipeline(drv::LogicalDevicePtr, drv::ComputePipelinePtr pipeline) {
    delete reinterpret_cast<CpuComputePipeline*>(pipeline);
    return true;
}
