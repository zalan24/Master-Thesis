#pragma once

namespace drv
{
struct CommandExecutionData;
struct CommandData;
struct TestData;
}  // namespace drv

namespace drv_cuda
{
bool transfer(const drv::CommandExecutionData* data, const drv::CommandData* command);
bool bind_compute_pipeline(const drv::CommandExecutionData* data, const drv::CommandData* command);
bool dispatch(const drv::CommandExecutionData* data, const drv::CommandData* command);
}  // namespace drv_cuda
