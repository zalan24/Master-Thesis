#pragma once

#include <string>
#include <thread>

void set_thread_name(std::thread* thread, const char* name);
std::string get_thread_name(std::thread::id id);
