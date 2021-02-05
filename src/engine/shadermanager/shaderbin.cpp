#include "shaderbin.h"

#include <cassert>
#include <fstream>
#include <stdexcept>

#include <binary_io.h>

ShaderBin::ShaderBin(const std::string& binfile) {
    std::ifstream in(binfile, std::ios::in | std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Could not open shader bin file: " + binfile);
    read(in);
}

static void write_configs(std::ostream& out, const ShaderBin::StageConfig& configs) {
    // vs
    write_string(out, configs.vs.entryPoint);
    // ps
    write_string(out, configs.ps.entryPoint);
    // cs
    write_string(out, configs.cs.entryPoint);
}

void ShaderBin::read(std::istream& in) {
    clear();
    throw std::runtime_error("Cannot read yet");
    if (!in.binary)
        throw std::runtime_error("Binary file expected for shaderBin input");
}

void ShaderBin::write(std::ostream& out) const {
    if (!out.binary)
        throw std::runtime_error("Binary file expected for shaderBin output");
    uint32_t shaderCount = static_cast<uint32_t>(shaders.size());
    write_data(out, shaderCount);
    for (const auto& [name, data] : shaders) {
        write_string(out, name);
        write_data(out, data.totalVariantCount);
        write_data(out, data.variantParamNum);
        write_data(out, data.variantValues);
        assert(data.stages.size() == data.totalVariantCount);
        for (uint32_t i = 0; i < data.totalVariantCount; ++i) {
            write_data(out, data.stages[i].stageOffsets);
            write_configs(out, data.stages[i].configs);
        }
        write_vector(out, data.codes);
    }
}

void ShaderBin::addShader(const std::string& name, ShaderData&& shader) {
    if (shaders.find(name) != shaders.end())
        throw std::runtime_error("A shader already exists with the given name: " + name);
    shaders[name] = std::move(shader);
}

void ShaderBin::clear() {
    shaders.clear();
}
