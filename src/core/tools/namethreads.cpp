#include "namethreads.h"

#include <map>
#include <mutex>
#include <sstream>
#include <thread>

// https://stackoverflow.com/questions/10121560/stdthread-naming-your-thread

static std::mutex mapMutex;
static std::map<std::thread::id, std::string> threadNames;

std::string get_thread_name(std::thread::id id) {
    std::unique_lock<std::mutex> lock(mapMutex);
    if (auto itr = threadNames.find(id); itr != threadNames.end())
        return itr->second;
    std::stringstream ss;
    ss << "#" << id;
    return ss.str();
}

#ifdef _WIN32
// because apparently vindoz can't event use lowercase header names
#    include <Windows.h>
const DWORD MS_VC_EXCEPTION = 0x406D1388;

#    pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType;      // Must be 0x1000.
    LPCSTR szName;     // Pointer to name (in user addr space).
    DWORD dwThreadID;  // Thread ID (-1=caller thread).
    DWORD dwFlags;     // Reserved for future use, must be zero.
} THREADNAME_INFO;
#    pragma pack(pop)

static void SetThreadName(uint32_t dwThreadID, const char* threadName) {
    // DWORD dwThreadID = ::GetThreadId( static_cast<HANDLE>( t.native_handle() ) );

    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags = 0;

    // __try {
    // WTF is this anyway??
    // RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR),
    //                reinterpret_cast<ULONG_PTR*>(&info));
    // }
    // __except (EXCEPTION_EXECUTE_HANDLER) {
    // }
}

void set_thread_name(std::thread* thread, const char* name) {
    DWORD threadId = ::GetThreadId(static_cast<HANDLE>(thread->native_handle()));
    SetThreadName(threadId, name);
    std::unique_lock<std::mutex> lock(mapMutex);
    threadNames[thread->get_id()] = std::string(name);
}

#elif defined(__linux__)
#    include <sys/prctl.h>
void set_thread_name(std::thread* thread, const char* name) {
    // prctl(PR_SET_NAME, threadName, 0, 0, 0);
}

#else
void set_thread_name(std::thread* thread, const char* name) {
    auto handle = thread->native_handle();
    pthread_setname_np(handle, name);
}
#endif
