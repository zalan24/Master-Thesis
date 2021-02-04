#include "spirvcompiler.h"

#include <SPIRV/GlslangToSpv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

Compiler::Compiler() {
    glslang::InitializeProcess();
}

Compiler::~Compiler() {
    glslang::FinalizeProcess();
}

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
        default:
            return EShLangVertex;
    }
}

static TBuiltInResource init_resources() {
    TBuiltInResource ret = {};
    ret.maxLights = 32;
    ret.maxClipPlanes = 6;
    ret.maxTextureUnits = 32;
    ret.maxTextureCoords = 32;
    ret.maxVertexAttribs = 64;
    ret.maxVertexUniformComponents = 4096;
    ret.maxVaryingFloats = 64;
    ret.maxVertexTextureImageUnits = 32;
    ret.maxCombinedTextureImageUnits = 80;
    ret.maxTextureImageUnits = 32;
    ret.maxFragmentUniformComponents = 4096;
    ret.maxDrawBuffers = 32;
    ret.maxVertexUniformVectors = 128;
    ret.maxVaryingVectors = 8;
    ret.maxFragmentUniformVectors = 16;
    ret.maxVertexOutputVectors = 16;
    ret.maxFragmentInputVectors = 15;
    ret.minProgramTexelOffset = -8;
    ret.maxProgramTexelOffset = 7;
    ret.maxClipDistances = 8;
    ret.maxComputeWorkGroupCountX = 65535;
    ret.maxComputeWorkGroupCountY = 65535;
    ret.maxComputeWorkGroupCountZ = 65535;
    ret.maxComputeWorkGroupSizeX = 1024;
    ret.maxComputeWorkGroupSizeY = 1024;
    ret.maxComputeWorkGroupSizeZ = 64;
    ret.maxComputeUniformComponents = 1024;
    ret.maxComputeTextureImageUnits = 16;
    ret.maxComputeImageUniforms = 8;
    ret.maxComputeAtomicCounters = 8;
    ret.maxComputeAtomicCounterBuffers = 1;
    ret.maxVaryingComponents = 60;
    ret.maxVertexOutputComponents = 64;
    ret.maxGeometryInputComponents = 64;
    ret.maxGeometryOutputComponents = 128;
    ret.maxFragmentInputComponents = 128;
    ret.maxImageUnits = 8;
    ret.maxCombinedImageUnitsAndFragmentOutputs = 8;
    ret.maxCombinedShaderOutputResources = 8;
    ret.maxImageSamples = 0;
    ret.maxVertexImageUniforms = 0;
    ret.maxTessControlImageUniforms = 0;
    ret.maxTessEvaluationImageUniforms = 0;
    ret.maxGeometryImageUniforms = 0;
    ret.maxFragmentImageUniforms = 8;
    ret.maxCombinedImageUniforms = 8;
    ret.maxGeometryTextureImageUnits = 16;
    ret.maxGeometryOutputVertices = 256;
    ret.maxGeometryTotalOutputComponents = 1024;
    ret.maxGeometryUniformComponents = 1024;
    ret.maxGeometryVaryingComponents = 64;
    ret.maxTessControlInputComponents = 128;
    ret.maxTessControlOutputComponents = 128;
    ret.maxTessControlTextureImageUnits = 16;
    ret.maxTessControlUniformComponents = 1024;
    ret.maxTessControlTotalOutputComponents = 4096;
    ret.maxTessEvaluationInputComponents = 128;
    ret.maxTessEvaluationOutputComponents = 128;
    ret.maxTessEvaluationTextureImageUnits = 16;
    ret.maxTessEvaluationUniformComponents = 1024;
    ret.maxTessPatchComponents = 120;
    ret.maxPatchVertices = 32;
    ret.maxTessGenLevel = 64;
    ret.maxViewports = 16;
    ret.maxVertexAtomicCounters = 0;
    ret.maxTessControlAtomicCounters = 0;
    ret.maxTessEvaluationAtomicCounters = 0;
    ret.maxGeometryAtomicCounters = 0;
    ret.maxFragmentAtomicCounters = 8;
    ret.maxCombinedAtomicCounters = 8;
    ret.maxAtomicCounterBindings = 1;
    ret.maxVertexAtomicCounterBuffers = 0;
    ret.maxTessControlAtomicCounterBuffers = 0;
    ret.maxTessEvaluationAtomicCounterBuffers = 0;
    ret.maxGeometryAtomicCounterBuffers = 0;
    ret.maxFragmentAtomicCounterBuffers = 1;
    ret.maxCombinedAtomicCounterBuffers = 1;
    ret.maxAtomicCounterBufferSize = 16384;
    ret.maxTransformFeedbackBuffers = 4;
    ret.maxTransformFeedbackInterleavedComponents = 64;
    ret.maxCullDistances = 8;
    ret.maxCombinedClipAndCullDistances = 8;
    ret.maxSamples = 4;
    ret.maxMeshOutputVerticesNV = 256;
    ret.maxMeshOutputPrimitivesNV = 512;
    ret.maxMeshWorkGroupSizeX_NV = 32;
    ret.maxMeshWorkGroupSizeY_NV = 1;
    ret.maxMeshWorkGroupSizeZ_NV = 1;
    ret.maxTaskWorkGroupSizeX_NV = 32;
    ret.maxTaskWorkGroupSizeY_NV = 1;
    ret.maxTaskWorkGroupSizeZ_NV = 1;
    ret.maxMeshViewCountNV = 4;
    ret.limits.nonInductiveForLoops = 1;
    ret.limits.whileLoops = 1;
    ret.limits.doWhileLoops = 1;
    ret.limits.generalUniformIndexing = 1;
    ret.limits.generalAttributeMatrixVectorIndexing = 1;
    ret.limits.generalVaryingIndexing = 1;
    ret.limits.generalSamplerIndexing = 1;
    ret.limits.generalVariableIndexing = 1;
    ret.limits.generalConstantMatrixVectorIndexing = 1;
    return ret;
}

std::vector<uint32_t> Compiler::GLSLtoSPV(const ShaderBin::Stage stage, const char* pShader) const {
    EShLanguage lang = find_language(stage);
    glslang::TShader shader(lang);
    glslang::TProgram program;
    const char* shaderStrings[1];
    TBuiltInResource resources = init_resources();

    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    shaderStrings[0] = pShader;
    shader.setStrings(shaderStrings, 1);

    if (!shader.parse(&resources, 100, false, messages)) {
        std::cerr << "Shader info log: " << std::endl << shader.getInfoLog() << std::endl;
        std::cerr << "Shader info debug log: " << std::endl
                  << shader.getInfoDebugLog() << std::endl;
        throw std::runtime_error("Could not compile shader");
    }

    program.addShader(&shader);

    //
    // Program-level processing...
    //

    if (!program.link(messages)) {
        std::cerr << "Shader info log: " << std::endl << shader.getInfoLog() << std::endl;
        std::cerr << "Shader info debug log: " << std::endl
                  << shader.getInfoDebugLog() << std::endl;
        throw std::runtime_error("Could not link shader");
    }

    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv);
    // TODO optimization
    return spirv;
}
