#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <serializable.h>

#include "material.h"
#include "pixel.h"
#include "resourcepool.h"
#include "texture.hpp"

class Mesh;

class MeshProvider
{
 public:
    MeshProvider();
    virtual ~MeshProvider();

    static MeshProvider* getSingleton() { return instance; }

    class CameraConfig final : public ISerializable
    {
     public:
        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;

        std::unordered_map<std::string, float> bones;
        glm::mat4 tm;
    };

    class ModelResource final : public ISerializable
    {
     public:
        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;

        std::string file;
        float size;
        std::string axisOrder;
        std::set<std::string> excludeMeshes;
        Material globalMaterialOverride;
        std::unordered_map<std::string, Material> materialOverrides;
        std::unordered_map<std::string, std::set<std::string>> meshSlots;
        CameraConfig cameraConfig;
    };

    using ModelResourceMap = std::unordered_map<std::string, ModelResource>;

    class ResourceDescriptor final : public ISerializable
    {
     public:
        ResourceDescriptor(const std::string& res_name);

        const std::string& getResourceName() const { return resName; }

        bool operator==(const ResourceDescriptor& other) const { return other.resName == resName; }

        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;

     private:
        std::string resName;
    };

    virtual GenericResourcePool::ResourceRef getResource(const ResourceDescriptor& desc) const = 0;
    virtual GenericResourcePool::ResourceRef createResource(const std::string& resName) const = 0;
    virtual GenericResourcePool::ResourceRef createResource(const Mesh& m) const = 0;
    virtual GenericResourcePool::ResourceRef createResource(const ModelResource& res) const = 0;

 private:
    static MeshProvider* instance;
};

namespace std
{
template <>
struct hash<MeshProvider::ResourceDescriptor>
{
    std::size_t operator()(MeshProvider::ResourceDescriptor const& s) const noexcept {
        return std::hash<std::string>{}(s.getResourceName());
    }
};
}  // namespace std
