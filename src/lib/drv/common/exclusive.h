#pragma once

#ifdef DEBUG
#    define EXCLUSIVE_TEST_ENABLED
#endif

#ifdef EXCLUSIVE_TEST_ENABLED
#    include <thread>
#endif

namespace drv
{
// Makes a class only usable on the owner thread
class Exclusive
{
 public:
    void setThreadOwnership();

 protected:
#ifdef EXCLUSIVE_TEST_ENABLED
    Exclusive();

    Exclusive(const Exclusive& other);
    Exclusive(Exclusive&& other);

    Exclusive& operator=(const Exclusive& other);
    Exclusive& operator=(Exclusive&& other);
#endif

    ~Exclusive() = default;

    bool checkThread() const;

 private:
#ifdef EXCLUSIVE_TEST_ENABLED
    std::thread::id id;
#endif
};
}  // namespace drv

#ifdef EXCLUSIVE_TEST_ENABLED
#    define CHECK_THREAD drv::drv_assert(checkThread(), "Object is bound to a different thread")
#else
#    define CHECK_THREAD static_cast<void>(nullptr)
#endif

#define SET_THREAD_OWNERSHIP setThreadOwnership()
