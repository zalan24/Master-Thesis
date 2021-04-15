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
    write_string(out, configs.vsEntryPoint);
    write_string(out, configs.psEntryPoint);
    write_string(out, configs.csEntryPoint);

    write_data(out, configs.polygonMode);
    write_data(out, configs.cullMode);
    write_data(out, configs.depthCompare);

    write_data(out, configs.useDepthClamp);
    write_data(out, configs.depthBiasEnable);
    write_data(out, configs.depthTest);
    write_data(out, configs.depthWrite);
    write_data(out, configs.stencilTest);
}

static void read_configs(std::istream& in, ShaderBin::StageConfig& configs) {
    read_string(in, configs.vsEntryPoint);
    read_string(in, configs.psEntryPoint);
    read_string(in, configs.csEntryPoint);

    read_data(in, configs.polygonMode);
    read_data(in, configs.cullMode);
    read_data(in, configs.depthCompare);

    read_data(in, configs.useDepthClamp);
    read_data(in, configs.depthBiasEnable);
    read_data(in, configs.depthTest);
    read_data(in, configs.depthWrite);
    read_data(in, configs.stencilTest);
}

void ShaderBin::read(std::istream& in) {
    clear();
    if (!in.binary)
        throw std::runtime_error("Binary file expected for shaderBin input");
    uint32_t header;
    read_data(in, header);
    if (header != FILE_HEADER)
        throw std::runtime_error("Invalid file header: " + std::to_string(header));
    uint32_t shaderCount = 0;
    read_data(in, shaderCount);
    for (size_t i = 0; i < shaderCount; ++i) {
        std::string name;
        read_string(in, name);
        ShaderData data;
        read_data(in, data.totalVariantCount);
        read_data(in, data.variantParamNum);
        read_data(in, data.variantValues);
        data.stages.resize(data.totalVariantCount);
        for (uint32_t j = 0; j < data.totalVariantCount; ++j) {
            read_data(in, data.stages[j].stageOffsets);
            read_data(in, data.stages[j].stageCodeSizes);
            read_configs(in, data.stages[j].configs);
        }
        read_vector(in, data.codes);
        shaders[name] = std::move(data);
    }
    uint32_t ending;
    read_data(in, ending);
    if (ending != FILE_END)
        throw std::runtime_error("Invalid file ending: " + std::to_string(ending));
}

void ShaderBin::write(std::ostream& out) const {
    if (!out.binary)
        throw std::runtime_error("Binary file expected for shaderBin output");
    write_data(out, FILE_HEADER);
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
            write_data(out, data.stages[i].stageCodeSizes);
            write_configs(out, data.stages[i].configs);
        }
        write_vector(out, data.codes);
    }
    write_data(out, FILE_END);
}

void ShaderBin::addShader(const std::string& name, ShaderData&& shader) {
    if (shaders.find(name) != shaders.end())
        throw std::runtime_error("A shader already exists with the given name: " + name);
    shaders[name] = std::move(shader);
}

void ShaderBin::clear() {
    shaders.clear();
}

const ShaderBin::ShaderData* ShaderBin::getShader(const std::string& name) const {
    auto itr = shaders.find(name);
    if (itr == shaders.end())
        return nullptr;
    return &itr->second;
}
