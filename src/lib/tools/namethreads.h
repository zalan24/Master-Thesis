#pragma once

namespace std
{
class thread;
}

void set_thread_name(std::thread* thread, const char* name);
