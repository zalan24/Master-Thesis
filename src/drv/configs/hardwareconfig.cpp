#include "hardwareconfig.h"

using namespace drv;

void DeviceLimits::writeJson(json& out) const {
    WRITE_OBJECT(maxPushConstantsSize, out);
}

void DeviceLimits::readJson(const json& in) {
    READ_OBJECT(maxPushConstantsSize, in);
}
