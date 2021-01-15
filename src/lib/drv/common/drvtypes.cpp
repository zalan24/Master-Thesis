#include "drvtypes.h"

using namespace drv;

CommandData::CommandData(CommandOptions_transfer transfer) : cmd(CMD_TRANSFER) {
    options.transfer = transfer;
}

CommandData::CommandData(CommandOptions_bind_compute_pipeline bindData)
  : cmd(CMD_BIND_COMPUTE_PIPELINE) {
    options.bindComputePipeline = bindData;
}

CommandData::CommandData(CommandOptions_bind_descriptor_sets bindInfo)
  : cmd(CMD_BIND_DESCRIPTOR_SETS) {
    options.bindDescriptorSets = bindInfo;
}

CommandData::CommandData(CommandOptions_dispatch dispatch) : cmd(CMD_DISPATCH) {
    options.dispatch = dispatch;
}
