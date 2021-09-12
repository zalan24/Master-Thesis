#pragma once

#include <serializable.h>
#include <filesystem>

namespace fs = std::filesystem;

void generate_capture_file(const fs::path& target, const ISerializable* captureObj, const std::string& screenShotFile);
