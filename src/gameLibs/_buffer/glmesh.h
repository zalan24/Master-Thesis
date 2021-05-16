#pragma once

#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <mesh.h>
#include <resourcepool.h>

#include "buffer.hpp"

class GlMesh
{
 public:
    struct BoneInfo
    {
        Mesh::BoneIndex parent;
        glm::mat4 defaultTm;
    };
    struct Material
    {
        GenericResourcePool::ResourceRef diffuseRef;
    };
    struct Segment
    {
        size_t vertexOffset;
        size_t indexOffset;
        size_t indexCount;
        std::vector<glm::mat4> boneOffsets;
        Mesh::MaterialIndex matId;
    };

    struct BoneState
    {
        glm::mat4 localTm;
    };

    struct State
    {
        Buffer<glm::mat4> bonesBuffer;
        mutable std::vector<glm::mat4> boneTms;  // temp storage for uploading
        std::vector<BoneState> bones;
        GLuint boneBinding;
    };

    GlMesh();

    void upload(const Mesh& mesh);

    size_t getSegmentCount() const;
    const Segment* getSegments() const;

    void bind() const;
    void unbind() const;

    void bindSkeleton() const;
    void unbindSkeleton() const;

    void bindBones(const State& state) const;
    void unbindBones(const State& state) const;

    State createState(GLuint boneBinding) const;
    void updateState(State& state) const;

    glm::mat4 getBoneWtm(const State& state, Mesh::BoneIndex boneId) const;

    void uploadBones(const State& state, const glm::mat4* offsets) const;

    const Material& getMat(Mesh::MaterialIndex id) const;

    void clear();

    size_t getSkeletonIndexCount() const { return boneIndexCount; }

    const Mesh::CameraData& getCameraData() const { return cameraData; }

 private:
    Buffer<Mesh::VertexData> glVertices;
    Buffer<Mesh::VertexIndex> glIndices;
    Buffer<Mesh::BoneVertex> glSkeletonVertices;
    Buffer<Mesh::VertexIndex> glSkeletonIndices;
    std::vector<Segment> segments;
    std::vector<BoneInfo> bones;
    std::vector<Material> materials;
    Mesh::CameraData cameraData;
    size_t boneIndexCount = 0;
};
