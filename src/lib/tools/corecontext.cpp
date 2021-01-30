#include "corecontext.h"

// R.I.P. header-only library

template <>
CoreContext* Singleton<CoreContext>::instance = nullptr;

CoreContext::CoreContext(const Config& config) : tempmem(config.tempmemSize) {
}
