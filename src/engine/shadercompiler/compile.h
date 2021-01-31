#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

bool compile_shader(const std::string& shaderFile,
                    const std::unordered_map<std::string, std::filesystem::path>& headerPaths);
