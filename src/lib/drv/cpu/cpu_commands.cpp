#include "cpu_commands.h"

#include <cstring>

#include <drverror.h>
#include <drvtypes.h>

#include "components/cpu_buffer.h"
#include "components/cpu_pipeline.h"
#include "components/cpu_shader_module.h"

bool drv_cpu::transfer(const drv::CommandData* command) {
    const drv::CommandOptions_transfer& options = command->options.transfer;
    for (unsigned int i = 0; i < options.numRegions; ++i) {
        Buffer* dst = reinterpret_cast<Buffer*>(options.dst);
        Buffer* src = reinterpret_cast<Buffer*>(options.src);
        memcpy(reinterpret_cast<uint8_t*>(dst->memory) + options.regions[i].dstOffset,
               reinterpret_cast<uint8_t*>(src->memory) + options.regions[i].srcOffset,
               options.regions[i].size);
    }
    return true;
}

static drv::CommandOptions_bind_compute_pipeline& getPipeline() {
    thread_local drv::CommandOptions_bind_compute_pipeline pipeline;
    return pipeline;
}

bool drv_cpu::bind_compute_pipeline(const drv::CommandData* command) {
    const drv::CommandOptions_bind_compute_pipeline& options = command->options.bindComputePipeline;
    getPipeline() = options;
    return true;
}

bool drv_cpu::bind_descriptor_sets(const drv::CommandData* command) {
    const drv::CommandOptions_bind_descriptor_sets& options = command->options.bindDescriptorSets;
    // TODO
    return false;
}

bool drv_cpu::dispatch(const drv::CommandData* command) {
    const drv::CommandOptions_dispatch& dispatch = command->options.dispatch;
    drv::CommandOptions_bind_compute_pipeline pipeline = getPipeline();
    const CpuComputePipeline* cpuPipeline =
      reinterpret_cast<const CpuComputePipeline*>(pipeline.pipeline);
    const ShaderModule* module = reinterpret_cast<const ShaderModule*>(cpuPipeline->computeModule);
#ifdef DEBUG
    drv::drv_assert(module->createInfo.info.compute.shaderFunction != nullptr,
                    "The shaderFunction is nullptr");
#endif

    ComputeShaderInput input{};  // TODO

    input.gridSizeX = dispatch.sizeX;
    input.gridSizeY = dispatch.sizeY;
    input.gridSizeZ = dispatch.sizeZ;

    module->createInfo.info.compute.shaderFunction(input);
    return true;
}
