#pragma once

#include <cstdint>

#include <serializable.h>

struct ShaderLimits final : public ISerializable
{
    bool nonInductiveForLoops;
    bool whileLoops;
    bool doWhileLoops;
    bool generalUniformIndexing;
    bool generalAttributeMatrixVectorIndexing;
    bool generalVaryingIndexing;
    bool generalSamplerIndexing;
    bool generalVariableIndexing;
    bool generalConstantMatrixVectorIndexing;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

struct ShaderBuiltInResource final : public ISerializable
{
    int maxLights;
    int maxClipPlanes;
    int maxTextureUnits;
    int maxTextureCoords;
    int maxVertexAttribs;
    int maxVertexUniformComponents;
    int maxVaryingFloats;
    int maxVertexTextureImageUnits;
    int maxCombinedTextureImageUnits;
    int maxTextureImageUnits;
    int maxFragmentUniformComponents;
    int maxDrawBuffers;
    int maxVertexUniformVectors;
    int maxVaryingVectors;
    int maxFragmentUniformVectors;
    int maxVertexOutputVectors;
    int maxFragmentInputVectors;
    int minProgramTexelOffset;
    int maxProgramTexelOffset;
    int maxClipDistances;
    int maxComputeWorkGroupCountX;
    int maxComputeWorkGroupCountY;
    int maxComputeWorkGroupCountZ;
    int maxComputeWorkGroupSizeX;
    int maxComputeWorkGroupSizeY;
    int maxComputeWorkGroupSizeZ;
    int maxComputeUniformComponents;
    int maxComputeTextureImageUnits;
    int maxComputeImageUniforms;
    int maxComputeAtomicCounters;
    int maxComputeAtomicCounterBuffers;
    int maxVaryingComponents;
    int maxVertexOutputComponents;
    int maxGeometryInputComponents;
    int maxGeometryOutputComponents;
    int maxFragmentInputComponents;
    int maxImageUnits;
    int maxCombinedImageUnitsAndFragmentOutputs;
    int maxCombinedShaderOutputResources;
    int maxImageSamples;
    int maxVertexImageUniforms;
    int maxTessControlImageUniforms;
    int maxTessEvaluationImageUniforms;
    int maxGeometryImageUniforms;
    int maxFragmentImageUniforms;
    int maxCombinedImageUniforms;
    int maxGeometryTextureImageUnits;
    int maxGeometryOutputVertices;
    int maxGeometryTotalOutputComponents;
    int maxGeometryUniformComponents;
    int maxGeometryVaryingComponents;
    int maxTessControlInputComponents;
    int maxTessControlOutputComponents;
    int maxTessControlTextureImageUnits;
    int maxTessControlUniformComponents;
    int maxTessControlTotalOutputComponents;
    int maxTessEvaluationInputComponents;
    int maxTessEvaluationOutputComponents;
    int maxTessEvaluationTextureImageUnits;
    int maxTessEvaluationUniformComponents;
    int maxTessPatchComponents;
    int maxPatchVertices;
    int maxTessGenLevel;
    int maxViewports;
    int maxVertexAtomicCounters;
    int maxTessControlAtomicCounters;
    int maxTessEvaluationAtomicCounters;
    int maxGeometryAtomicCounters;
    int maxFragmentAtomicCounters;
    int maxCombinedAtomicCounters;
    int maxAtomicCounterBindings;
    int maxVertexAtomicCounterBuffers;
    int maxTessControlAtomicCounterBuffers;
    int maxTessEvaluationAtomicCounterBuffers;
    int maxGeometryAtomicCounterBuffers;
    int maxFragmentAtomicCounterBuffers;
    int maxCombinedAtomicCounterBuffers;
    int maxAtomicCounterBufferSize;
    int maxTransformFeedbackBuffers;
    int maxTransformFeedbackInterleavedComponents;
    int maxCullDistances;
    int maxCombinedClipAndCullDistances;
    int maxSamples;
    int maxMeshOutputVerticesNV;
    int maxMeshOutputPrimitivesNV;
    int maxMeshWorkGroupSizeX_NV;
    int maxMeshWorkGroupSizeY_NV;
    int maxMeshWorkGroupSizeZ_NV;
    int maxTaskWorkGroupSizeX_NV;
    int maxTaskWorkGroupSizeY_NV;
    int maxTaskWorkGroupSizeZ_NV;
    int maxMeshViewCountNV;
    // int maxDualSourceDrawBuffersEXT;

    ShaderLimits limits;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};

struct CompileOptions final : public ISerializable
{
    ShaderBuiltInResource shaderResources;

    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};
