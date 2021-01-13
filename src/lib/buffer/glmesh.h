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
        Mesh::MaterialIndex matId;
    };

    struct BoneState
    {
        glm::mat4 localTm;
    };

    struct State
    {
        Buffer<glm::mat4> bonesBuffer;
        std::vector<glm::mat4> boneTms;
        std::vector<BoneState> bones;
        bool invalidBones = true;
        GLuint boneBinding;
    };

    GlMesh();

    void upload(const Mesh& mesh);

    size_t getSegmentCount() const;
    const Segment* getSegments() const;

    void bind() const;
    void unbind() const;

    void bindState(const State& state) const;
    void unbindState(const State& state) const;

    State createState(GLuint boneBinding) const;
    void updateState(State& state) const;

    const Material& getMat(Mesh::MaterialIndex id) const;

    void clear();

 private:
    Buffer<Mesh::VertexData> glBuffer;
    Buffer<Mesh::VertexIndex> glIndices;
    std::vector<Segment> segments;
    std::vector<BoneInfo> bones;
    std::vector<Material> materials;
};
