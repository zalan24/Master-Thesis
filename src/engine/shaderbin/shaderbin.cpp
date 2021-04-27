#include "shaderbin.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <binary_io.h>

ShaderBin::ShaderBin(drv::DeviceLimits _limits) : limits(_limits) {
}

ShaderBin::ShaderBin(const std::string& binfile) {
    std::ifstream in(binfile, std::ios::in | std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Could not open shader bin file: " + binfile);
    read(in);
}

// void ShaderBin::StageConfig::writeJson(json& out) const {
// }

// void ShaderBin::StageConfig::readJson(const json& in) {
// }

void ShaderBin::StageConfig::write(std::ostream& out) const {
    for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
        write_string(out, entryPoints[i]);

    write_data(out, polygonMode);
    write_data(out, cullMode);
    write_data(out, depthCompare);

    write_data(out, useDepthClamp);
    write_data(out, depthBiasEnable);
    write_data(out, depthTest);
    write_data(out, depthWrite);
    write_data(out, stencilTest);

    uint32_t count = static_cast<uint32_t>(attachments.size());
    write_data(out, count);
    for (uint32_t i = 0; i < count; ++i) {
        write_string(out, attachments[i].name);
        write_data(out, attachments[i].info);
        write_data(out, attachments[i].location);
    }
}

void ShaderBin::StageConfig::read(std::istream& in) {
    for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
        read_string(in, entryPoints[i]);

    read_data(in, polygonMode);
    read_data(in, cullMode);
    read_data(in, depthCompare);

    read_data(in, useDepthClamp);
    read_data(in, depthBiasEnable);
    read_data(in, depthTest);
    read_data(in, depthWrite);
    read_data(in, stencilTest);

    uint32_t count = 0;
    read_data(in, count);
    attachments.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        read_string(in, attachments[i].name);
        read_data(in, attachments[i].info);
        read_data(in, attachments[i].location);
    }
}

void ShaderBin::read(std::istream& in) {
    clear();
    if (!in.binary)
        throw std::runtime_error("Binary file expected for shaderBin input");
    uint32_t header;
    read_data(in, header);
    if (header != FILE_HEADER)
        throw std::runtime_error("Invalid file header: " + std::to_string(header));
    std::string stamp;
    read_string(in, stamp);
    if (stamp != __TIMESTAMP__)
        throw std::runtime_error("Mismatching timestamps for shader binary: " + stamp
                                 + " (shader)   !=   " + std::string(__TIMESTAMP__) + " (current)");
    uint64_t hash;
    read_data(in, hash);
    // TODO
    // if (hash !=)
    //     throw std::runtime_error("Incompatible shader binary");
    uint32_t shaderCount = 0;
    read_data(in, shaderCount);
    std::string limitsStr;
    read_string(in, limitsStr);
    {
        std::stringstream ss(limitsStr);
        limits.read(ss);
    }
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
            data.stages[j].configs.read(in);
            // TODO;
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
    write_string(out, __TIMESTAMP__);
    write_data(out, shaderHeadersHash);
    uint32_t shaderCount = static_cast<uint32_t>(shaders.size());
    write_data(out, shaderCount);
    {
        std::stringstream ss;
        limits.write(ss);
        write_string(out, ss.str());
    }
    for (const auto& [name, data] : shaders) {
        write_string(out, name);
        write_data(out, data.totalVariantCount);
        write_data(out, data.variantParamNum);
        write_data(out, data.variantValues);
        assert(data.stages.size() == data.totalVariantCount);
        for (uint32_t i = 0; i < data.totalVariantCount; ++i) {
            write_data(out, data.stages[i].stageOffsets);
            write_data(out, data.stages[i].stageCodeSizes);
            data.stages[i].configs.write(out);
            // TODO;
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
