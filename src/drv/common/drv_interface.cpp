#include "drv_interface.h"

using namespace drv;

const static std::pair<StateTrackingConfig::Verbosity, const char*> verbosityNames[] = {
  {StateTrackingConfig::SILENT_FIXES, "SilentFixes"},
  {StateTrackingConfig::DEBUG_ERRORS, "DebugErrors"},
  {StateTrackingConfig::ALL_ERRORS, "AllErrors"}};

void StateTrackingConfig::writeJson(json& out) const {
    WRITE_ENUM(verbosity, out, verbosityNames);
    WRITE_OBJECT(immediateBarriers, out);
    // WRITE_OBJECT(immediateEventBarriers, out);
    WRITE_OBJECT(forceAllDstStages, out);
    WRITE_OBJECT(forceFlush, out);
    WRITE_OBJECT(forceInvalidateAll, out);
    WRITE_OBJECT(syncAllOperations, out);
}

void StateTrackingConfig::readJson(const json& in) {
    Config defaultConfig;
    READ_ENUM_OPT(verbosity, in, defaultConfig.verbosity, verbosityNames);
    READ_OBJECT_OPT(immediateBarriers, in, defaultConfig.immediateBarriers);
    // READ_OBJECT_OPT(immediateEventBarriers, in, defaultConfig.immediateEventBarriers);
    READ_OBJECT_OPT(forceAllDstStages, in, defaultConfig.forceAllDstStages);
    READ_OBJECT_OPT(forceFlush, in, defaultConfig.forceFlush);
    READ_OBJECT_OPT(forceInvalidateAll, in, defaultConfig.forceInvalidateAll);
    READ_OBJECT_OPT(syncAllOperations, in, defaultConfig.syncAllOperations);
}
