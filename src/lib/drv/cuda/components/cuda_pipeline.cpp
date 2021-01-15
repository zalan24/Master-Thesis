#include "drvcuda.h"

#include <drverror.h>

drv::PipelineLayoutPtr drv_cuda::create_pipeline_layout(drv::LogicalDevicePtr device,
                                                        const drv::PipelineLayoutCreateInfo* info) {
    drv::drv_assert(false, "Not implemented");
    return drv::NULL_HANDLE;
}

bool drv_cuda::destroy_pipeline_layout(drv::LogicalDevicePtr device,
                                       drv::PipelineLayoutPtr layout) {
    drv::drv_assert(false, "Not implemented");
    return false;
}

bool drv_cuda::create_compute_pipeline(drv::LogicalDevicePtr device, unsigned int count,
                                       const drv::ComputePipelineCreateInfo* infos,
                                       drv::ComputePipelinePtr* pipelines) {
    drv::drv_assert(false, "Not implemented");
    return false;
}

bool drv_cuda::destroy_compute_pipeline(drv::LogicalDevicePtr device,
                                        drv::ComputePipelinePtr pipeline) {
    drv::drv_assert(false, "Not implemented");
    return false;
}
