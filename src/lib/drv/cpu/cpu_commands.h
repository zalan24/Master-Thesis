#pragma once

namespace drv
{
struct CommandOptions_transfer;
struct CommandData;
struct TestData;
}  // namespace drv

namespace drv_cpu
{
bool transfer(const drv::CommandData* command);
bool bind_compute_pipeline(const drv::CommandData* command);
bool bind_descriptor_sets(const drv::CommandData* command);
bool dispatch(const drv::CommandData* command);
}  // namespace drv_cpu
