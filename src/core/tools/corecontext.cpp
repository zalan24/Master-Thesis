#include "corecontext.h"

template <>
CoreContext* Singleton<CoreContext>::instance = nullptr;

CoreContext::CoreContext(const Config& config)
  : tempmem(safe_cast<StackMemory::size_t>(config.tempmemSize)) {
}
