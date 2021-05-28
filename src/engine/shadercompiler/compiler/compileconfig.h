#pragma once

#include <cstdint>

#include <serializable.h>

struct ShaderLimits final : public IAutoSerializable<ShaderLimits>
{
    REFLECTABLE
    (
        (bool) nonInductiveForLoops,
        (bool) whileLoops,
        (bool) doWhileLoops,
        (bool) generalUniformIndexing,
        (bool) generalAttributeMatrixVectorIndexing,
        (bool) generalVaryingIndexing,
        (bool) generalSamplerIndexing,
        (bool) generalVariableIndexing,
        (bool) generalConstantMatrixVectorIndexing
    )

    ShaderLimits()
      : nonInductiveForLoops(true),
        whileLoops(true),
        doWhileLoops(true),
        generalUniformIndexing(true),
        generalAttributeMatrixVectorIndexing(true),
        generalVaryingIndexing(true),
        generalSamplerIndexing(true),
        generalVariableIndexing(true),
        generalConstantMatrixVectorIndexing(true) {}

    // bool nonInductiveForLoops;
    // bool whileLoops;
    // bool doWhileLoops;
    // bool generalUniformIndexing;
    // bool generalAttributeMatrixVectorIndexing;
    // bool generalVaryingIndexing;
    // bool generalSamplerIndexing;
    // bool generalVariableIndexing;
    // bool generalConstantMatrixVectorIndexing;

    // REFLECT()
};

struct ShaderBuiltInResource final : public IAutoSerializable<ShaderBuiltInResource>
{
    struct Misc final : public IAutoSerializable<Misc>
    {
        REFLECTABLE
        (
            (int) maxLights,
            (int) maxClipPlanes,
            (int) maxTextureUnits,
            (int) maxTextureCoords,
            (int) maxVaryingFloats,
            (int) maxCombinedTextureImageUnits,
            (int) maxTextureImageUnits,
            (int) maxDrawBuffers,
            (int) maxVaryingVectors,
            (int) minProgramTexelOffset,
            (int) maxProgramTexelOffset,
            (int) maxClipDistances,
            (int) maxVaryingComponents,
            (int) maxImageUnits,
            (int) maxCombinedShaderOutputResources,
            (int) maxImageSamples,
            (int) maxCombinedImageUniforms,
            (int) maxViewports,
            (int) maxCombinedAtomicCounters,
            (int) maxAtomicCounterBindings,
            (int) maxCombinedAtomicCounterBuffers,
            (int) maxAtomicCounterBufferSize,
            (int) maxTransformFeedbackBuffers,
            (int) maxTransformFeedbackInterleavedComponents,
            (int) maxCullDistances,
            (int) maxCombinedClipAndCullDistances,
            (int) maxSamples
        )

        Misc()
          : maxLights(32),
            maxClipPlanes(6),
            maxTextureUnits(32),
            maxTextureCoords(32),
            maxVaryingFloats(64),
            maxCombinedTextureImageUnits(80),
            maxTextureImageUnits(32),
            maxDrawBuffers(32),
            maxVaryingVectors(8),
            minProgramTexelOffset(-8),
            maxProgramTexelOffset(7),
            maxClipDistances(8),
            maxVaryingComponents(60),
            maxImageUnits(8),
            maxCombinedShaderOutputResources(8),
            maxImageSamples(0),
            maxCombinedImageUniforms(8),
            maxViewports(16),
            maxCombinedAtomicCounters(8),
            maxAtomicCounterBindings(1),
            maxCombinedAtomicCounterBuffers(1),
            maxAtomicCounterBufferSize(16384),
            maxTransformFeedbackBuffers(4),
            maxTransformFeedbackInterleavedComponents(64),
            maxCullDistances(8),
            maxCombinedClipAndCullDistances(8),
            maxSamples(4) {}
    };
    struct VertexShader final : public IAutoSerializable<VertexShader>
    {
        REFLECTABLE
        (
            (int) maxVertexAttribs,
            (int) maxVertexUniformComponents,
            (int) maxVertexTextureImageUnits,
            (int) maxVertexUniformVectors,
            (int) maxVertexOutputVectors,
            (int) maxVertexOutputComponents,
            (int) maxVertexImageUniforms,
            (int) maxVertexAtomicCounters,
            (int) maxVertexAtomicCounterBuffers
        )

        VertexShader()
          : maxVertexAttribs(64),
            maxVertexUniformComponents(4096),
            maxVertexTextureImageUnits(32),
            maxVertexUniformVectors(128),
            maxVertexOutputVectors(16),
            maxVertexOutputComponents(64),
            maxVertexImageUniforms(0),
            maxVertexAtomicCounters(0),
            maxVertexAtomicCounterBuffers(0) {}
    };
    struct GeometryShader final : public IAutoSerializable<GeometryShader>
    {
        REFLECTABLE
        (
            (int) maxGeometryInputComponents,
            (int) maxGeometryOutputComponents,
            (int) maxGeometryImageUniforms,
            (int) maxGeometryTextureImageUnits,
            (int) maxGeometryOutputVertices,
            (int) maxGeometryTotalOutputComponents,
            (int) maxGeometryUniformComponents,
            (int) maxGeometryVaryingComponents,
            (int) maxGeometryAtomicCounters,
            (int) maxGeometryAtomicCounterBuffers
        )

        GeometryShader()
          : maxGeometryInputComponents(64),
            maxGeometryOutputComponents(128),
            maxGeometryImageUniforms(0),
            maxGeometryTextureImageUnits(16),
            maxGeometryOutputVertices(256),
            maxGeometryTotalOutputComponents(1024),
            maxGeometryUniformComponents(1024),
            maxGeometryVaryingComponents(64),
            maxGeometryAtomicCounters(0),
            maxGeometryAtomicCounterBuffers(0) {}
    };
    struct TessControlShader final : public IAutoSerializable<TessControlShader>
    {
        REFLECTABLE
        (
            (int) maxTessControlImageUniforms,
            (int) maxTessControlInputComponents,
            (int) maxTessControlOutputComponents,
            (int) maxTessControlTextureImageUnits,
            (int) maxTessControlUniformComponents,
            (int) maxTessControlTotalOutputComponents,
            (int) maxTessControlAtomicCounters,
            (int) maxTessControlAtomicCounterBuffers,
            (int) maxTessPatchComponents,
            (int) maxPatchVertices,
            (int) maxTessGenLevel
        )

        TessControlShader()
          : maxTessControlImageUniforms(0),
            maxTessControlInputComponents(128),
            maxTessControlOutputComponents(128),
            maxTessControlTextureImageUnits(16),
            maxTessControlUniformComponents(1024),
            maxTessControlTotalOutputComponents(4096),
            maxTessControlAtomicCounters(0),
            maxTessControlAtomicCounterBuffers(0),
            maxTessPatchComponents(120),
            maxPatchVertices(32),
            maxTessGenLevel(64) {}
    };
    struct TessEvalShader final : public IAutoSerializable<TessEvalShader>
    {
        REFLECTABLE
        (
            (int) maxTessEvaluationImageUniforms,
            (int) maxTessEvaluationInputComponents,
            (int) maxTessEvaluationOutputComponents,
            (int) maxTessEvaluationTextureImageUnits,
            (int) maxTessEvaluationUniformComponents,
            (int) maxTessEvaluationAtomicCounters,
            (int) maxTessEvaluationAtomicCounterBuffers
        )

        TessEvalShader()
          : maxTessEvaluationImageUniforms(0),
            maxTessEvaluationInputComponents(128),
            maxTessEvaluationOutputComponents(128),
            maxTessEvaluationTextureImageUnits(16),
            maxTessEvaluationUniformComponents(1024),
            maxTessEvaluationAtomicCounters(0),
            maxTessEvaluationAtomicCounterBuffers(0) {}
    };
    struct FragmentShader final : public IAutoSerializable<FragmentShader>
    {
        REFLECTABLE
        (
            (int) maxFragmentUniformComponents,
            (int) maxFragmentUniformVectors,
            (int) maxFragmentInputVectors,
            (int) maxFragmentInputComponents,
            (int) maxCombinedImageUnitsAndFragmentOutputs,
            (int) maxFragmentImageUniforms,
            (int) maxFragmentAtomicCounters,
            (int) maxFragmentAtomicCounterBuffers
        )

        FragmentShader()
          : maxFragmentUniformComponents(4096),
            maxFragmentUniformVectors(16),
            maxFragmentInputVectors(15),
            maxFragmentInputComponents(128),
            maxCombinedImageUnitsAndFragmentOutputs(8),
            maxFragmentImageUniforms(8),
            maxFragmentAtomicCounters(8),
            maxFragmentAtomicCounterBuffers(1) {}
    };
    struct ComputeShader final : public IAutoSerializable<ComputeShader>
    {
        REFLECTABLE
        (
            (int) maxComputeWorkGroupCountX,
            (int) maxComputeWorkGroupCountY,
            (int) maxComputeWorkGroupCountZ,
            (int) maxComputeWorkGroupSizeX,
            (int) maxComputeWorkGroupSizeY,
            (int) maxComputeWorkGroupSizeZ,
            (int) maxComputeUniformComponents,
            (int) maxComputeTextureImageUnits,
            (int) maxComputeImageUniforms,
            (int) maxComputeAtomicCounters,
            (int) maxComputeAtomicCounterBuffers
        )

        ComputeShader()
          : maxComputeWorkGroupCountX(65535),
            maxComputeWorkGroupCountY(65535),
            maxComputeWorkGroupCountZ(65535),
            maxComputeWorkGroupSizeX(1024),
            maxComputeWorkGroupSizeY(1024),
            maxComputeWorkGroupSizeZ(64),
            maxComputeUniformComponents(1024),
            maxComputeTextureImageUnits(16),
            maxComputeImageUniforms(8),
            maxComputeAtomicCounters(8),
            maxComputeAtomicCounterBuffers(1) {}
    };
    struct MeshShader final : public IAutoSerializable<MeshShader>
    {
        REFLECTABLE
        (
            (int) maxMeshOutputVerticesNV,
            (int) maxMeshOutputPrimitivesNV,
            (int) maxMeshWorkGroupSizeX_NV,
            (int) maxMeshWorkGroupSizeY_NV,
            (int) maxMeshWorkGroupSizeZ_NV,
            (int) maxMeshViewCountNV
        )

        MeshShader()
          : maxMeshOutputVerticesNV(256),
            maxMeshOutputPrimitivesNV(512),
            maxMeshWorkGroupSizeX_NV(32),
            maxMeshWorkGroupSizeY_NV(1),
            maxMeshWorkGroupSizeZ_NV(1),
            maxMeshViewCountNV(4) {}
    };
    struct TaskShader final : public IAutoSerializable<TaskShader>
    {
        REFLECTABLE
        (
            (int) maxTaskWorkGroupSizeX_NV,
            (int) maxTaskWorkGroupSizeY_NV,
            (int) maxTaskWorkGroupSizeZ_NV
        )

        TaskShader()
          : maxTaskWorkGroupSizeX_NV(32),
            maxTaskWorkGroupSizeY_NV(1),
            maxTaskWorkGroupSizeZ_NV(1) {}
    };
    REFLECTABLE
    (
        (Misc) misc,
        (VertexShader) vs,
        (GeometryShader) gs,
        (TessControlShader) tc,
        (TessEvalShader) te,
        (FragmentShader) ps,
        (ComputeShader) cs,
        (MeshShader) ms,
        (TaskShader) ts,
        (ShaderLimits) limits
    )

    // int maxLights;
    // int maxClipPlanes;
    // int maxTextureUnits;
    // int maxTextureCoords;
    // int maxVertexAttribs;
    // int maxVertexUniformComponents;
    // int maxVaryingFloats;
    // int maxVertexTextureImageUnits;
    // int maxCombinedTextureImageUnits;
    // int maxTextureImageUnits;
    // int maxFragmentUniformComponents;
    // int maxDrawBuffers;
    // int maxVertexUniformVectors;
    // int maxVaryingVectors;
    // int maxFragmentUniformVectors;
    // int maxVertexOutputVectors;
    // int maxFragmentInputVectors;
    // int minProgramTexelOffset;
    // int maxProgramTexelOffset;
    // int maxClipDistances;
    // int maxComputeWorkGroupCountX;
    // int maxComputeWorkGroupCountY;
    // int maxComputeWorkGroupCountZ;
    // int maxComputeWorkGroupSizeX;
    // int maxComputeWorkGroupSizeY;
    // int maxComputeWorkGroupSizeZ;
    // int maxComputeUniformComponents;
    // int maxComputeTextureImageUnits;
    // int maxComputeImageUniforms;
    // int maxComputeAtomicCounters;
    // int maxComputeAtomicCounterBuffers;
    // int maxVaryingComponents;
    // int maxVertexOutputComponents;
    // int maxGeometryInputComponents;
    // int maxGeometryOutputComponents;
    // int maxFragmentInputComponents;
    // int maxImageUnits;
    // int maxCombinedImageUnitsAndFragmentOutputs;
    // int maxCombinedShaderOutputResources;
    // int maxImageSamples;
    // int maxVertexImageUniforms;
    // int maxTessControlImageUniforms;
    // int maxTessEvaluationImageUniforms;
    // int maxGeometryImageUniforms;
    // int maxFragmentImageUniforms;
    // int maxCombinedImageUniforms;
    // int maxGeometryTextureImageUnits;
    // int maxGeometryOutputVertices;
    // int maxGeometryTotalOutputComponents;
    // int maxGeometryUniformComponents;
    // int maxGeometryVaryingComponents;
    // int maxTessControlInputComponents;
    // int maxTessControlOutputComponents;
    // int maxTessControlTextureImageUnits;
    // int maxTessControlUniformComponents;
    // int maxTessControlTotalOutputComponents;
    // int maxTessEvaluationInputComponents;
    // int maxTessEvaluationOutputComponents;
    // int maxTessEvaluationTextureImageUnits;
    // int maxTessEvaluationUniformComponents;
    // int maxTessPatchComponents;
    // int maxPatchVertices;
    // int maxTessGenLevel;
    // int maxViewports;
    // int maxVertexAtomicCounters;
    // int maxTessControlAtomicCounters;
    // int maxTessEvaluationAtomicCounters;
    // int maxGeometryAtomicCounters;
    // int maxFragmentAtomicCounters;
    // int maxCombinedAtomicCounters;
    // int maxAtomicCounterBindings;
    // int maxVertexAtomicCounterBuffers;
    // int maxTessControlAtomicCounterBuffers;
    // int maxTessEvaluationAtomicCounterBuffers;
    // int maxGeometryAtomicCounterBuffers;
    // int maxFragmentAtomicCounterBuffers;
    // int maxCombinedAtomicCounterBuffers;
    // int maxAtomicCounterBufferSize;
    // int maxTransformFeedbackBuffers;
    // int maxTransformFeedbackInterleavedComponents;
    // int maxCullDistances;
    // int maxCombinedClipAndCullDistances;
    // int maxSamples;
    // int maxMeshOutputVerticesNV;
    // int maxMeshOutputPrimitivesNV;
    // int maxMeshWorkGroupSizeX_NV;
    // int maxMeshWorkGroupSizeY_NV;
    // int maxMeshWorkGroupSizeZ_NV;
    // int maxTaskWorkGroupSizeX_NV;
    // int maxTaskWorkGroupSizeY_NV;
    // int maxTaskWorkGroupSizeZ_NV;
    // int maxMeshViewCountNV;
    // // int maxDualSourceDrawBuffersEXT;

    // ShaderLimits limits;

    // REFLECT()
};

struct CompileOptions final : public IAutoSerializable<CompileOptions>
{
    REFLECTABLE
    (
        (ShaderBuiltInResource) shaderResources
    )

    // ShaderBuiltInResource shaderResources;

    // REFLECT()
};
