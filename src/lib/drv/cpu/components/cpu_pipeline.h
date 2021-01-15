#include "drvcpu.h"

namespace drv_cpu
{
struct CpuPipelineLayout
{};
struct CpuComputePipeline
{
    drv::ShaderModulePtr computeModule;
};
}  // namespace drv_cpu
