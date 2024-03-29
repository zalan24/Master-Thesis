#include "shaderbin.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <binary_io.h>

ShaderBin::ShaderBin(drv::DeviceLimits _limits)
  : shaderHeadersHash(0), limits(_limits), codeLen(0) {
}

ShaderBin::ShaderBin(const fs::path& binfile) {
    importFromFile(binfile);
}

// void ShaderBin::StageConfig::writeJson(json& out) const {
// }

// void ShaderBin::StageConfig::readJson(const json& in) {
// }

// void ShaderBin::StageConfig::write(std::ostream& out) const {
//     for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
//         write_string(out, entryPoints[i]);

//     write_data(out, polygonMode);
//     write_data(out, cullMode);
//     write_data(out, depthCompare);

//     write_data(out, useDepthClamp);
//     write_data(out, depthBiasEnable);
//     write_data(out, depthTest);
//     write_data(out, depthWrite);
//     write_data(out, stencilTest);

//     uint32_t count = static_cast<uint32_t>(attachments.size());
//     write_data(out, count);
//     for (uint32_t i = 0; i < count; ++i) {
//         write_string(out, attachments[i].name);
//         write_data(out, attachments[i].info);
//         write_data(out, attachments[i].location);
//     }
// }

// void ShaderBin::StageConfig::read(std::istream& in) {
//     for (uint32_t i = 0; i < ShaderBin::NUM_STAGES; ++i)
//         read_string(in, entryPoints[i]);

//     read_data(in, polygonMode);
//     read_data(in, cullMode);
//     read_data(in, depthCompare);

//     read_data(in, useDepthClamp);
//     read_data(in, depthBiasEnable);
//     read_data(in, depthTest);
//     read_data(in, depthWrite);
//     read_data(in, stencilTest);

//     uint32_t count = 0;
//     read_data(in, count);
//     attachments.resize(count);
//     for (uint32_t i = 0; i < count; ++i) {
//         read_string(in, attachments[i].name);
//         read_data(in, attachments[i].info);
//         read_data(in, attachments[i].location);
//     }
// }

// void ShaderBin::read(std::istream& in) {
//     clear();
//     if (!in.binary)
//         throw std::runtime_error("Binary file expected for shaderBin input");
//     uint32_t header;
//     read_data(in, header);
//     if (header != FILE_HEADER)
//         throw std::runtime_error("Invalid file header: " + std::to_string(header));
//     std::string stamp;
//     read_string(in, stamp);
//     if (stamp != __TIMESTAMP__)
//         throw std::runtime_error("Mismatching timestamps for shader binary: " + stamp
//                                  + " (shader)   !=   " + std::string(__TIMESTAMP__) + " (current)");
//     uint64_t hash;
//     read_data(in, hash);
//     // TODO
//     // if (hash !=)
//     //     throw std::runtime_error("Incompatible shader binary");
//     uint32_t shaderCount = 0;
//     read_data(in, shaderCount);
//     std::string limitsStr;
//     read_string(in, limitsStr);
//     {
//         std::stringstream ss(limitsStr);
//         limits.read(ss);
//     }
//     for (size_t i = 0; i < shaderCount; ++i) {
//         std::string name;
//         read_string(in, name);
//         ShaderData data;
//         read_data(in, data.totalVariantCount);
//         // read_data(in, data.variantParamNum);
//         // read_data(in, data.variantValues);
//         data.stages.resize(data.totalVariantCount);
//         for (uint32_t j = 0; j < data.totalVariantCount; ++j) {
//             read_data(in, data.stages[j].stageOffsets);
//             read_data(in, data.stages[j].stageCodeSizes);
//             data.stages[j].configs.read(in);
//             // TODO;
//         }
//         shaders[name] = std::move(data);
//     }
//     uint32_t codesStart;
//     read_data(in, codesStart);
//     if (codesStart != FILE_CODES_START)
//         throw std::runtime_error("Invalid code block start marker: " + std::to_string(codesStart));
//     uint64_t size;
//     read_data(in, size);
//     codeBlocks.resize(size);
//     for (uint64_t i = 0; i < size; ++i) {
//         uint32_t blockStart;
//         read_data(in, blockStart);
//         if (blockStart != FILE_BLOCK_START)
//             throw std::runtime_error("Invalid block start: " + std::to_string(blockStart));
//         read_vector(in, codeBlocks[i]);
//         uint32_t blockEnd;
//         read_data(in, blockEnd);
//         if (blockEnd != FILE_BLOCK_END)
//             throw std::runtime_error("Invalid block ending: " + std::to_string(blockEnd));
//     }
//     uint32_t ending;
//     read_data(in, ending);
//     if (ending != FILE_END)
//         throw std::runtime_error("Invalid file ending: " + std::to_string(ending));
//     if (in.fail())
//         throw std::runtime_error("Could not read shader binary from input stream");
// }

// void ShaderBin::write(std::ostream& out) const {
//     if (!out.binary)
//         throw std::runtime_error("Binary file expected for shaderBin output");
//     write_data(out, FILE_HEADER);
//     write_string(out, __TIMESTAMP__);
//     write_data(out, shaderHeadersHash);
//     uint32_t shaderCount = static_cast<uint32_t>(shaders.size());
//     write_data(out, shaderCount);
//     {
//         std::stringstream ss;
//         limits.write(ss);
//         write_string(out, ss.str());
//     }
//     for (const auto& [name, data] : shaders) {
//         write_string(out, name);
//         write_data(out, data.totalVariantCount);
//         // write_data(out, data.variantParamNum);
//         // write_data(out, data.variantValues);
//         assert(data.stages.size() == data.totalVariantCount);
//         for (uint32_t i = 0; i < data.totalVariantCount; ++i) {
//             write_data(out, data.stages[i].stageOffsets);
//             write_data(out, data.stages[i].stageCodeSizes);
//             data.stages[i].configs.write(out);
//             // TODO;
//         }
//     }
//     write_data(out, FILE_CODES_START);
//     uint64_t codeBlocksSize = codeBlocks.size();
//     write_data(out, codeBlocksSize);
//     for (size_t i = 0; i < codeBlocks.size(); ++i) {
//         write_data(out, FILE_BLOCK_START);
//         write_vector(out, codeBlocks[i]);
//         write_data(out, FILE_BLOCK_END);
//     }
//     write_data(out, FILE_END);
//     if (out.fail())
//         throw std::runtime_error("Could not write shader binary to output stream");
// }

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

uint32_t* ShaderBin::getCode(uint64_t ind) {
    return &codeBlocks[high_index(ind)][low_index(ind)];
}

const uint32_t* ShaderBin::getCode(uint64_t ind) const {
    return &codeBlocks[high_index(ind)][low_index(ind)];
}

uint64_t ShaderBin::addShaderCode(size_t len, const uint32_t* code) {
    if (len > CODE_BLOCK_LOW_FILETR)
        throw std::runtime_error("Shader code doesn't fit in a code block");
    if (low_index(codeLen) + len >= CODE_BLOCK_SIZE || high_index(codeLen) >= codeBlocks.size()) {
        codeLen = codeBlocks.size() << CODE_BLOCK_BITS;
        codeBlocks.resize(codeBlocks.size() + 1);
        codeBlocks.back().resize(CODE_BLOCK_SIZE);
    }
    uint64_t ret = codeLen;
    std::copy(code, code + len, getCode(ret));
    codeLen += len;
    return ret;
}
