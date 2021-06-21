#include "statscache.h"

#include <binary_io.h>

#include <drverror.h>

StatsCache::StatsCache() {
    semaphore.set(0);
}
