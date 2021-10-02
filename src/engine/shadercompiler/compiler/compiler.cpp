#include "compiler.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <iterator>
#include <set>

#include <binary_io.h>

#include <glslang/SPIRV/GlslangToSpv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

namespace fs = std::filesystem;

static EShLanguage find_language(const ShaderBin::Stage stage) {
    switch (stage) {
        case ShaderBin::Stage::VS:
            return EShLangVertex;
        // case  ShaderBin::Stage::TS:
        //     return EShLangTessControl;
        // case  ShaderBin::Stage::TE:
        //     return EShLangTessEvaluation;
        // case  ShaderBin::Stage::GS:
        //     return EShLangGeometry;
        case ShaderBin::Stage::PS:
            return EShLangFragment;
        case ShaderBin::Stage::CS:
            return EShLangCompute;
        case ShaderBin::Stage::NUM_STAGES:
            throw std::runtime_error("Unhandled shader stage");
    }
}

Compiler::Compiler() {
    glslang::InitializeProcess();
}

Compiler::~Compiler() {
    glslang::FinalizeProcess();
}

void Compiler::addShaders(PreprocessorData&& data) {
    for (const auto& header : data.headers)
        if (headerToCollection.find(header.first) != headerToCollection.end())
            throw std::runtime_error("Two headers exist with the same name: " + header.first);
    for (const auto& shader : data.sources)
        if (shaderToCollection.find(shader.first) != shaderToCollection.end())
            throw std::runtime_error("Two shaders exists with the same name: " + shader.first);
    size_t id = collections.size();
    for (const auto& header : data.headers)
        headerToCollection[header.first] = id;
    for (const auto& shader : data.sources)
        shaderToCollection[shader.first] = id;
    collections.push_back(std::move(data));
}

void Compiler::generateShaders(const GenerateOptions& options, const fs::path& dir,
                               const std::vector<std::string>& shaderNames) {
    std::set<std::string> compiledShaders;
    for (const auto& shader : shaderNames)
        compiledShaders.insert(shader);
    if (fs::exists(dir)) {
        for (const auto& p : fs::directory_iterator(dir)) {
            const std::string name = p.path().stem().string();
            if (shaderToCollection.find(name) == shaderToCollection.end())
                std::cout << "Removing unused shader dir: " << name << " (" << p.path() << ")\n";
        }
    }
    else
        fs::create_directories(dir);
    const std::string optionsHash = options.hash();
    bool recompileAll = false;
    if (cache.options.hash() != optionsHash) {
        recompileAll = true;
        cache.options = options;
    }
    std::map<std::string, fs::path> hashToFile[ShaderBin::NUM_STAGES];
    for (const auto& [name, id] : shaderToCollection) {
        if (compiledShaders.size() > 0 && compiledShaders.count(name) == 0)
            continue;
        const fs::path genFolder = getGenFolder(dir, name);
        const size_t collectionId = shaderToCollection.find(name)->second;
        const PreprocessorData& prepData = collections[collectionId];
        const ShaderObjectData& objData = prepData.sources.find(name)->second;
        const std::string shaderHash = objData.hash();
        std::string includesHash = "";
        for (const auto& h : objData.allIncludes)
            includesHash +=
              collections[headerToCollection.find(h)->second].headers.find(h)->second.hash();
        includesHash = hash_string(includesHash);
        if (!recompileAll && fs::exists(genFolder)) {
            if (auto itr = cache.shaders.find(name); itr != cache.shaders.end()) {
                if (itr->second.includesHash == includesHash
                    && itr->second.shaderHash == shaderHash)
                    continue;
            }
        }
        ShaderCache& shaderCache = cache.shaders[name];
        shaderCache.shaderHash = shaderHash;
        shaderCache.includesHash = includesHash;
        std::cout << "Generating shader: " << name << std::endl;
        if (!fs::exists(genFolder))
            fs::create_directories(genFolder);
        std::vector<std::vector<fs::path>> codePath;
        codePath.resize(objData.variantCount);
        if (shaderCache.codeHashes.size() != objData.variantCount)
            shaderCache.codeHashes.resize(objData.variantCount);
        for (uint32_t i = 0; i < objData.variantCount; ++i) {
            ShaderGenerationInput genInput;
            const ShaderObjectData::ComputeUnit cu = objData.readComputeUnite(&genInput);
            codePath[i].resize(ShaderBin::NUM_STAGES);
            for (uint32_t j = 0; j < ShaderBin::NUM_STAGES; ++j) {
                std::stringstream code;
                if (!generateShaderCode(objData, cu, genInput, i, static_cast<ShaderBin::Stage>(j),
                                        code))
                    continue;
                const std::string codeHash = hash_string(code.str());
                const std::string stageName =
                  ShaderBin::get_stage_name(static_cast<ShaderBin::Stage>(j));
                const fs::path shaderPath =
                  getGlslPath(dir, name, i, static_cast<ShaderBin::Stage>(j));
                codePath[i][j] = shaderPath;
                // std::cout << "Glsl hash: " << name << " " << i << " " << stageName << ":\t"
                //           << codeHash << std::endl;
                if (codeHash != shaderCache.codeHashes[i][stageName] || !fs::exists(shaderPath)) {
                    std::ofstream out(shaderPath.c_str());
                    if (!out.is_open())
                        throw std::runtime_error("Could not open file: " + shaderPath.string());
                    out << code.str();
                    shaderCache.codeHashes[i][stageName] = codeHash;
                }
            }
        }
        std::cout << "Compiling shader: " << name << std::endl;
        if (shaderCache.binaryConfigHashes.size() != objData.variantCount)
            shaderCache.binaryConfigHashes.resize(objData.variantCount);
        for (uint32_t i = 0; i < objData.variantCount; ++i) {
            for (uint32_t j = 0; j < ShaderBin::NUM_STAGES; ++j) {
                ShaderBin::Stage stage = static_cast<ShaderBin::Stage>(j);
                fs::path spvPath = getSpvPath(dir, name, i, stage);
                if (codePath[i][j].empty()) {
                    if (fs::exists(spvPath))
                        fs::remove(spvPath);
                    continue;
                }
                const std::string stageName = ShaderBin::get_stage_name(stage);
                std::string binaryHash = hash_string(shaderCache.codeHashes[i][stageName] + ".."
                                                     + options.compileOptions.hash() + ".."
                                                     + options.limits.hash() + ".." + stageName);
                // std::cout << "Hash: " << name << " " << i << " " << stageName << ":\t" << binaryHash
                //           << "\t\t" << shaderCache.codeHashes[i][stageName] << std::endl;
                if (auto itr = shaderCache.binaryConfigHashes[i].find(stageName);
                    itr != shaderCache.binaryConfigHashes[i].end() && itr->second == binaryHash
                    && fs::exists(spvPath))
                    continue;
                if (auto itr = hashToFile[stage].find(binaryHash); itr != hashToFile[stage].end()) {
                    if (fs::exists(spvPath))
                        fs::remove(spvPath);
                    fs::copy_file(itr->second, spvPath);
                    shaderCache.binaryConfigHashes[i][stageName] = binaryHash;
                    continue;
                }
                compile_shader(options.compileOptions, options.limits, stage, codePath[i][j],
                               spvPath);
                hashToFile[stage][binaryHash] = spvPath;
                // std::cout << "Write hash " << i << " " << stageName << ": " << binaryHash
                //           << std::endl;
                shaderCache.binaryConfigHashes[i][stageName] = binaryHash;
            }
        }
    }
}

static void generate_shader_code(std::ostream& out) {
    out << "#version 450\n";
    out << "#extension GL_ARB_separate_shader_objects : enable\n";
    if constexpr (featureconfig::params.shaderPrint)
        out << "#extension GL_EXT_debug_printf : enable\n";
    out << "\n";
}

static void generate_shader_attachments(const ShaderBin::StageConfig& configs, std::ostream& code) {
    for (const auto& attachment : configs.attachments) {
        code << "layout(location = " << static_cast<uint32_t>(attachment.location) << ") ";
        if (attachment.info & ShaderBin::AttachmentInfo::WRITE)
            code << "out ";
        else
            throw std::runtime_error("Implement input attachments as well");
        // TODO handle depth attachment
        code << "vec4 " << attachment.name << ";\n";
    }
    code << "\n";
}

static TBuiltInResource reade_resources(const ShaderBuiltInResource& resources) {
    TBuiltInResource ret;
    ret.maxLights = resources.misc.maxLights;
    ret.maxClipPlanes = resources.misc.maxClipPlanes;
    ret.maxTextureUnits = resources.misc.maxTextureUnits;
    ret.maxTextureCoords = resources.misc.maxTextureCoords;
    ret.maxVertexAttribs = resources.vs.maxVertexAttribs;
    ret.maxVertexUniformComponents = resources.vs.maxVertexUniformComponents;
    ret.maxVaryingFloats = resources.misc.maxVaryingFloats;
    ret.maxVertexTextureImageUnits = resources.vs.maxVertexTextureImageUnits;
    ret.maxCombinedTextureImageUnits = resources.misc.maxCombinedTextureImageUnits;
    ret.maxTextureImageUnits = resources.misc.maxTextureImageUnits;
    ret.maxFragmentUniformComponents = resources.ps.maxFragmentUniformComponents;
    ret.maxDrawBuffers = resources.misc.maxDrawBuffers;
    ret.maxVertexUniformVectors = resources.vs.maxVertexUniformVectors;
    ret.maxVaryingVectors = resources.misc.maxVaryingVectors;
    ret.maxFragmentUniformVectors = resources.ps.maxFragmentUniformVectors;
    ret.maxVertexOutputVectors = resources.vs.maxVertexOutputVectors;
    ret.maxFragmentInputVectors = resources.ps.maxFragmentInputVectors;
    ret.minProgramTexelOffset = resources.misc.minProgramTexelOffset;
    ret.maxProgramTexelOffset = resources.misc.maxProgramTexelOffset;
    ret.maxClipDistances = resources.misc.maxClipDistances;
    ret.maxComputeWorkGroupCountX = resources.cs.maxComputeWorkGroupCountX;
    ret.maxComputeWorkGroupCountY = resources.cs.maxComputeWorkGroupCountY;
    ret.maxComputeWorkGroupCountZ = resources.cs.maxComputeWorkGroupCountZ;
    ret.maxComputeWorkGroupSizeX = resources.cs.maxComputeWorkGroupSizeX;
    ret.maxComputeWorkGroupSizeY = resources.cs.maxComputeWorkGroupSizeY;
    ret.maxComputeWorkGroupSizeZ = resources.cs.maxComputeWorkGroupSizeZ;
    ret.maxComputeUniformComponents = resources.cs.maxComputeUniformComponents;
    ret.maxComputeTextureImageUnits = resources.cs.maxComputeTextureImageUnits;
    ret.maxComputeImageUniforms = resources.cs.maxComputeImageUniforms;
    ret.maxComputeAtomicCounters = resources.cs.maxComputeAtomicCounters;
    ret.maxComputeAtomicCounterBuffers = resources.cs.maxComputeAtomicCounterBuffers;
    ret.maxVaryingComponents = resources.misc.maxVaryingComponents;
    ret.maxVertexOutputComponents = resources.vs.maxVertexOutputComponents;
    ret.maxGeometryInputComponents = resources.gs.maxGeometryInputComponents;
    ret.maxGeometryOutputComponents = resources.gs.maxGeometryOutputComponents;
    ret.maxFragmentInputComponents = resources.ps.maxFragmentInputComponents;
    ret.maxImageUnits = resources.misc.maxImageUnits;
    ret.maxCombinedImageUnitsAndFragmentOutputs =
      resources.ps.maxCombinedImageUnitsAndFragmentOutputs;
    ret.maxCombinedShaderOutputResources = resources.misc.maxCombinedShaderOutputResources;
    ret.maxImageSamples = resources.misc.maxImageSamples;
    ret.maxVertexImageUniforms = resources.vs.maxVertexImageUniforms;
    ret.maxTessControlImageUniforms = resources.tc.maxTessControlImageUniforms;
    ret.maxTessEvaluationImageUniforms = resources.te.maxTessEvaluationImageUniforms;
    ret.maxGeometryImageUniforms = resources.gs.maxGeometryImageUniforms;
    ret.maxFragmentImageUniforms = resources.ps.maxFragmentImageUniforms;
    ret.maxCombinedImageUniforms = resources.misc.maxCombinedImageUniforms;
    ret.maxGeometryTextureImageUnits = resources.gs.maxGeometryTextureImageUnits;
    ret.maxGeometryOutputVertices = resources.gs.maxGeometryOutputVertices;
    ret.maxGeometryTotalOutputComponents = resources.gs.maxGeometryTotalOutputComponents;
    ret.maxGeometryUniformComponents = resources.gs.maxGeometryUniformComponents;
    ret.maxGeometryVaryingComponents = resources.gs.maxGeometryVaryingComponents;
    ret.maxTessControlInputComponents = resources.tc.maxTessControlInputComponents;
    ret.maxTessControlOutputComponents = resources.tc.maxTessControlOutputComponents;
    ret.maxTessControlTextureImageUnits = resources.tc.maxTessControlTextureImageUnits;
    ret.maxTessControlUniformComponents = resources.tc.maxTessControlUniformComponents;
    ret.maxTessControlTotalOutputComponents = resources.tc.maxTessControlTotalOutputComponents;
    ret.maxTessEvaluationInputComponents = resources.te.maxTessEvaluationInputComponents;
    ret.maxTessEvaluationOutputComponents = resources.te.maxTessEvaluationOutputComponents;
    ret.maxTessEvaluationTextureImageUnits = resources.te.maxTessEvaluationTextureImageUnits;
    ret.maxTessEvaluationUniformComponents = resources.te.maxTessEvaluationUniformComponents;
    ret.maxTessPatchComponents = resources.tc.maxTessPatchComponents;
    ret.maxPatchVertices = resources.tc.maxPatchVertices;
    ret.maxTessGenLevel = resources.tc.maxTessGenLevel;
    ret.maxViewports = resources.misc.maxViewports;
    ret.maxVertexAtomicCounters = resources.vs.maxVertexAtomicCounters;
    ret.maxTessControlAtomicCounters = resources.tc.maxTessControlAtomicCounters;
    ret.maxTessEvaluationAtomicCounters = resources.te.maxTessEvaluationAtomicCounters;
    ret.maxGeometryAtomicCounters = resources.gs.maxGeometryAtomicCounters;
    ret.maxFragmentAtomicCounters = resources.ps.maxFragmentAtomicCounters;
    ret.maxCombinedAtomicCounters = resources.misc.maxCombinedAtomicCounters;
    ret.maxAtomicCounterBindings = resources.misc.maxAtomicCounterBindings;
    ret.maxVertexAtomicCounterBuffers = resources.vs.maxVertexAtomicCounterBuffers;
    ret.maxTessControlAtomicCounterBuffers = resources.tc.maxTessControlAtomicCounterBuffers;
    ret.maxTessEvaluationAtomicCounterBuffers = resources.te.maxTessEvaluationAtomicCounterBuffers;
    ret.maxGeometryAtomicCounterBuffers = resources.gs.maxGeometryAtomicCounterBuffers;
    ret.maxFragmentAtomicCounterBuffers = resources.ps.maxFragmentAtomicCounterBuffers;
    ret.maxCombinedAtomicCounterBuffers = resources.misc.maxCombinedAtomicCounterBuffers;
    ret.maxAtomicCounterBufferSize = resources.misc.maxAtomicCounterBufferSize;
    ret.maxTransformFeedbackBuffers = resources.misc.maxTransformFeedbackBuffers;
    ret.maxTransformFeedbackInterleavedComponents =
      resources.misc.maxTransformFeedbackInterleavedComponents;
    ret.maxCullDistances = resources.misc.maxCullDistances;
    ret.maxCombinedClipAndCullDistances = resources.misc.maxCombinedClipAndCullDistances;
    ret.maxSamples = resources.misc.maxSamples;
    ret.maxMeshOutputVerticesNV = resources.ms.maxMeshOutputVerticesNV;
    ret.maxMeshOutputPrimitivesNV = resources.ms.maxMeshOutputPrimitivesNV;
    ret.maxMeshWorkGroupSizeX_NV = resources.ms.maxMeshWorkGroupSizeX_NV;
    ret.maxMeshWorkGroupSizeY_NV = resources.ms.maxMeshWorkGroupSizeY_NV;
    ret.maxMeshWorkGroupSizeZ_NV = resources.ms.maxMeshWorkGroupSizeZ_NV;
    ret.maxTaskWorkGroupSizeX_NV = resources.ts.maxTaskWorkGroupSizeX_NV;
    ret.maxTaskWorkGroupSizeY_NV = resources.ts.maxTaskWorkGroupSizeY_NV;
    ret.maxTaskWorkGroupSizeZ_NV = resources.ts.maxTaskWorkGroupSizeZ_NV;
    ret.maxMeshViewCountNV = resources.ms.maxMeshViewCountNV;
    // ret.maxDualSourceDrawBuffersEXT = resources.maxDualSourceDrawBuffersEXT;

    ret.limits.nonInductiveForLoops = resources.limits.nonInductiveForLoops;
    ret.limits.whileLoops = resources.limits.whileLoops;
    ret.limits.doWhileLoops = resources.limits.doWhileLoops;
    ret.limits.generalUniformIndexing = resources.limits.generalUniformIndexing;
    ret.limits.generalAttributeMatrixVectorIndexing =
      resources.limits.generalAttributeMatrixVectorIndexing;
    ret.limits.generalVaryingIndexing = resources.limits.generalVaryingIndexing;
    ret.limits.generalSamplerIndexing = resources.limits.generalSamplerIndexing;
    ret.limits.generalVariableIndexing = resources.limits.generalVariableIndexing;
    ret.limits.generalConstantMatrixVectorIndexing =
      resources.limits.generalConstantMatrixVectorIndexing;
    return ret;
}

void Compiler::compile_shader(const CompileOptions& compileOptions, const drv::DeviceLimits& limits,
                              ShaderBin::Stage stage, const fs::path& glsl, const fs::path& spv) {
    EShLanguage lang = find_language(stage);
    glslang::TShader shader(lang);
    glslang::TProgram program;
    const char* shaderStrings[1];
    TBuiltInResource resources = reade_resources(compileOptions.shaderResources);

    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);

    std::ifstream shaderIn(glsl.c_str());
    if (!shaderIn.is_open())
        throw std::runtime_error("Could not open own file for reading: " + glsl.string());
    std::string pShader((std::istreambuf_iterator<char>(shaderIn)),
                        std::istreambuf_iterator<char>());
    shaderIn.close();

    shaderStrings[0] = pShader.c_str();
    shader.setStrings(shaderStrings, 1);

    if (!shader.parse(&resources, 100, false, messages)) {
        std::cerr << "Shader info log:" << std::endl << shader.getInfoLog() << std::endl;
        std::cerr << "Shader info debug log:" << std::endl << shader.getInfoDebugLog() << std::endl;
        throw std::runtime_error("Could not compile shader: " + glsl.string());
    }

    program.addShader(&shader);

    //
    // Program-level processing...
    //

    if (!program.link(messages)) {
        std::cerr << "Shader info log:" << std::endl << shader.getInfoLog() << std::endl;
        std::cerr << "Shader info debug log:" << std::endl << shader.getInfoDebugLog() << std::endl;
        throw std::runtime_error("Could not link shader: " + glsl.string());
    }

    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv);
    // TODO optimization

    std::ofstream out(spv.c_str(), std::ios::binary | std::ios::out);
    if (!out.is_open())
        throw std::runtime_error("Could not open file: " + spv.string());
    write_vector(out, spirv);
}

bool Compiler::generateShaderCode(const ShaderObjectData& objData,
                                  const ShaderObjectData::ComputeUnit& cu,
                                  const ShaderGenerationInput& genInput, uint32_t variantId,
                                  ShaderBin::Stage stage, std::ostream& out) const {
    generate_shader_code(out);
    struct PushConstInfo
    {
        PushConstEntry entry;
        uint32_t headerOffset;
    };
    std::vector<PushConstInfo> pushConsts;

    if (stage == ShaderBin::Stage::CS) {
        for (const auto& [header, infos] : objData.headerToConfigToResinfosCompute) {
            const auto& incData =
              collections[headerToCollection.find(header)->second].headers.find(header)->second;
            uint32_t localVariant =
              (variantId / objData.headerVariantIdMultiplier.find(header)->second)
              % incData.totalVariantMultiplier;
            uint32_t configId = objData.variantToConfigId[variantId];
            uint32_t structId = incData.localVariantToStructIdCompute[localVariant];
            if (structId == INVALID_STRUCT_ID)
                continue;
            for (PushConstEntry pushConst : incData.structIdToGlslStructDesc[structId]) {
                pushConsts.push_back({pushConst, infos[configId].pushConstOffset});
            }
        }
    }
    else {
        for (const auto& [header, infos] : objData.headerToConfigToResinfosGraphics) {
            const auto& incData =
              collections[headerToCollection.find(header)->second].headers.find(header)->second;
            uint32_t localVariant =
              (variantId / objData.headerVariantIdMultiplier.find(header)->second)
              % incData.totalVariantMultiplier;
            uint32_t configId = objData.variantToConfigId[variantId];
            uint32_t structId = incData.localVariantToStructIdGraphics[localVariant];
            if (structId == INVALID_STRUCT_ID)
                continue;
            for (PushConstEntry pushConst : incData.structIdToGlslStructDesc[structId]) {
                pushConsts.push_back({pushConst, infos[configId].pushConstOffset});
            }
        }
    }
    if (!pushConsts.empty()) {
        std::sort(pushConsts.begin(), pushConsts.end(),
                  [](const PushConstInfo& lhs, const PushConstInfo& rhs) {
                      return lhs.entry.localOffset + lhs.headerOffset
                             < rhs.entry.localOffset + rhs.headerOffset;
                  });
        out << "layout(std430, push_constant) uniform pushConstants {\n";
        for (const auto& itr : pushConsts)
            out << "    layout(offset=" << (itr.entry.localOffset + itr.headerOffset) << ") "
                << itr.entry.type << " " << itr.entry.name << "; // (" << itr.headerOffset << "+"
                << itr.entry.localOffset << ")\n";
        out << "} PushConstants;\n";
    }

    PipelineResourceUsage resourceUsage;
    ShaderBin::StageConfig cfg =
      read_stage_configs(objData.resources, variantId, objData.variants,
                         objData.variantIdMultiplier, genInput, resourceUsage);
    if (cfg.entryPoints[stage] == "")
        return false;
    if (stage == ShaderBin::PS)
        generate_shader_attachments(cfg, out);
    out << format_variant(variantId, objData.variants, cu.stages[stage],
                          objData.variantIdMultiplier);
    return true;
}

ShaderBin Compiler::link(const fs::path& shaderDir, const drv::DeviceLimits& limits) const {
    std::cout << "Linking shader binary..." << std::endl;
    ShaderBin ret(limits);
    std::unordered_map<std::string, std::pair<size_t, size_t>> spvHashToOffset;
    for (const auto& collection : collections) {
        for (const auto& [name, objData] : collection.sources) {
            auto shaderCacheItr = cache.shaders.find(name);
            if (shaderCacheItr == cache.shaders.end())
                throw std::runtime_error("Linker cannot find shader info in cache: " + name);
            const ShaderCache& shaderCache = shaderCacheItr->second;
            if (shaderCache.binaryConfigHashes.size() != objData.variantCount)
                throw std::runtime_error("Shader cache is not up-to-date for: " + name);
            const ShaderGenerationInput genInput = objData.readGenInput();
            ShaderBin::ShaderData shaderData;
            shaderData.totalVariantCount = objData.variantCount;
            shaderData.stages.reserve(shaderData.totalVariantCount);
            for (uint32_t i = 0; i < shaderData.totalVariantCount; ++i) {
                PipelineResourceUsage resourceUsage;
                ShaderBin::ShaderData::StageData stageData;
                stageData.configs =
                  read_stage_configs(objData.resources, i, objData.variants,
                                     objData.variantIdMultiplier, genInput, resourceUsage);
                for (uint32_t j = 0; j < ShaderBin::NUM_STAGES; ++j) {
                    if (stageData.configs.entryPoints[j] == "") {
                        stageData.stageOffsets[j] = ShaderBin::ShaderData::INVALID_SHADER;
                        stageData.stageCodeSizes[j] = 0;
                        continue;
                    }
                    const std::string stageName =
                      ShaderBin::get_stage_name(static_cast<ShaderBin::Stage>(j));
                    auto hashItr = shaderCache.binaryConfigHashes[i].find(stageName);
                    if (hashItr == shaderCache.binaryConfigHashes[i].end())
                        throw std::runtime_error("Shader cache is not up-to-date for: " + name);
                    std::string hash = hashItr->second;
                    if (auto itr = spvHashToOffset.find(hash); itr != spvHashToOffset.end()) {
                        stageData.stageOffsets[j] = itr->second.first;
                        stageData.stageCodeSizes[j] = itr->second.second;
                    }
                    else {
                        const fs::path spvPath =
                          getSpvPath(shaderDir, name, i, static_cast<ShaderBin::Stage>(j));
                        std::ifstream in(spvPath.c_str(), std::ios::binary | std::ios::in);
                        if (!in.is_open())
                            throw std::runtime_error("Could not open spv file: "
                                                     + spvPath.string());

                        std::vector<uint32_t> spirv;
                        read_vector(in, spirv);
                        if (spirv.size() == 0)
                            throw std::runtime_error("Invalid spv file: " + spvPath.string());
                        uint64_t offset = ret.addShaderCode(spirv.size(), spirv.data());
                        stageData.stageOffsets[j] = offset;
                        stageData.stageCodeSizes[j] = spirv.size();
                        // std::cout << "Compile new shader: " << name << " " << i << " " << stageName
                        //           << " (" << offset << ", " << spirv.size() << ")   " << hash
                        //           << std::endl;
                        spvHashToOffset[hash] =
                          std::make_pair<size_t, size_t>(size_t(offset), spirv.size());
                    }
                    assert(stageData.stageCodeSizes[j] != 0);
                }
                shaderData.stages.push_back(std::move(stageData));
            }
            ret.addShader(name, std::move(shaderData));
        }
    }
    return ret;
}

fs::path Compiler::getGenFolder(const fs::path& parentDir, const std::string& name) const {
    return parentDir / fs::path{name};
}

fs::path Compiler::getGlslPath(const fs::path& parentDir, const std::string& name,
                               uint32_t variantId, ShaderBin::Stage stage) const {
    const std::string stageName = ShaderBin::get_stage_name(stage);
    return getGenFolder(parentDir, name)
           / fs::path{name + std::to_string(variantId) + "." + stageName};
}
fs::path Compiler::getSpvPath(const fs::path& parentDir, const std::string& name,
                              uint32_t variantId, ShaderBin::Stage stage) const {
    fs::path ret = getGlslPath(parentDir, name, variantId, stage);
    ret += ".spv";
    return ret;
}
