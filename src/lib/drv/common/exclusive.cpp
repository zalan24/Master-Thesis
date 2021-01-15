#include "exclusive.h"

using namespace drv;

#ifdef EXCLUSIVE_TEST_ENABLED

Exclusive::Exclusive() : id(std::this_thread::get_id()) {
}

Exclusive::Exclusive(const Exclusive&) : id(std::this_thread::get_id()) {
}

Exclusive::Exclusive(Exclusive&&) : id(std::this_thread::get_id()) {
}

Exclusive& Exclusive::operator=(const Exclusive&) {
    id = std::this_thread::get_id();
    return *this;
}

Exclusive& Exclusive::operator=(Exclusive&&) {
    id = std::this_thread::get_id();
    return *this;
}

#endif

bool Exclusive::checkThread() const {
#ifdef EXCLUSIVE_TEST_ENABLED
    return id == std::this_thread::get_id();
#else
    return true;
#endif
}

void Exclusive::setThreadOwnership() {
#ifdef EXCLUSIVE_TEST_ENABLED
    id = std::this_thread::get_id();
#endif
}
