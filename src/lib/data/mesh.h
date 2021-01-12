#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include "material.h"

class Mesh
{
 public:
    using VertexIndex = uint32_t;
    using BoneIndex = uint32_t;
    using MaterialIndex = uint32_t;
    using SegmentIndex = uint32_t;

    static constexpr MaterialIndex INVALID_MATERIAL = std::numeric_limits<MaterialIndex>::max();
    static constexpr BoneIndex INVALID_BONE = std::numeric_limits<BoneIndex>::max();
    static constexpr size_t MAX_BONES = 4;

    struct Bone
    {
        BoneIndex parent;
        glm::mat4 localTm;
        glm::mat4 offset;  // TODO what's this used for?
    };
    class Skeleton
    {
     public:
        Skeleton();

        BoneIndex addBone(Bone&& bone);
        void registerBone(BoneIndex boneId, const std::string& name);

        BoneIndex getRoot() const { return rootBone; }

     private:
        std::unordered_map<std::string, BoneIndex> boneMap;
        std::vector<Bone> bones;
        BoneIndex rootBone;
    };
    struct VertexData
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;
        glm::vec2 texcoord;
        glm::ivec4 boneIds;
        glm::vec4 boneWeights;
        VertexData(const glm::vec3& _position, const glm::vec3& _normal,
                   const glm::vec3& _color = glm::vec3(0, 0, 0),
                   const glm::vec2& _texcoord = glm::vec2(0, 0),
                   const glm::ivec4& _boneIds = glm::ivec4(0, 0, 0, 0),
                   const glm::vec4& _boneWeights = glm::vec4(1, 0, 0, 0))
          : position(_position),
            normal(_normal),
            color(_color),
            texcoord(_texcoord),
            boneIds(_boneIds),
            boneWeights(_boneWeights) {}
    };

    struct Segment
    {
        Mesh::MaterialIndex mat = INVALID_MATERIAL;
        std::vector<Mesh::VertexData> vertices;
        std::vector<Mesh::VertexIndex> indices;

        Mesh::VertexIndex addVertex(const Mesh::VertexData& vert);
        void addFace();     // adds last three vertices
        void addFaceRev();  // adds last three vertices flipped
        void addFace(Mesh::VertexIndex p1, Mesh::VertexIndex p2, Mesh::VertexIndex p3);
        void setNodeTm(const glm::mat4& tm);
        void addChild(const Mesh& m);
        void addChild(Mesh&& m);
    };

    Mesh();

    MaterialIndex addMaterial(Material&& mat);
    MaterialIndex addMaterial(const Material& mat);

    SegmentIndex addSegment(Segment&& segment);
    void sortSegments();

    size_t getSegmentCount() const { return segments.size(); }
    const Segment& getSegment(size_t id) const { return segments[id]; }

    const Material* getMaterial(MaterialIndex mat) const;

    const Skeleton* getSkeleton() const { return &skeleton; }
    Skeleton* getSkeleton() { return &skeleton; }

 private:
    std::vector<Material> materials;
    std::vector<Segment> segments;
    Skeleton skeleton;
};
