#include "corecontext.h"

template <>
CoreContext* Singleton<CoreContext>::instance = nullptr;

CoreContext::CoreContext(const Config& config) : tempmem(config.tempmemSize) {
}
