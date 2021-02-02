#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class BlockFile;

struct Variants
{
    std::unordered_map<std::string, std::vector<std::string>> values;
};

bool read_variants(const BlockFile* blockFile, Variants& variants);

bool compile_shader(const std::string& shaderFile,
                    const std::unordered_map<std::string, std::filesystem::path>& headerPaths);

bool generate_header(const std::string& shaderFile, const std::string& outputFolder);

bool generate_binary(const std::vector<Variants>& variants,
                     std::vector<std::vector<uint32_t>>& binary, std::istream& shader);
