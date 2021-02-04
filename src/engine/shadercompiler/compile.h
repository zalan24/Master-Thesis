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

#include <shaderbin.h>

class BlockFile;

struct Variants
{
    std::unordered_map<std::string, std::vector<std::string>> values;
};

struct VariantConfig
{
    std::unordered_map<std::string, size_t> variantValues;
};

struct ShaderGenerationInput
{
    std::stringstream vs;
    std::stringstream vsCfg;
    std::stringstream ps;
    std::stringstream psCfg;
    std::stringstream cs;
    std::stringstream csCfg;
};

bool read_variants(const BlockFile* blockFile, Variants& variants);

bool compile_shader(ShaderBin& shaderBin, const std::string& shaderFile,
                    const std::unordered_map<std::string, std::filesystem::path>& headerPaths);

bool generate_header(const std::string& shaderFile, const std::string& outputFolder);

bool generate_binary(ShaderBin::ShaderData& shaderData, const std::vector<Variants>& variants,
                     ShaderGenerationInput&& input);

VariantConfig get_variant_config(size_t index, const std::vector<Variants>& variants);
