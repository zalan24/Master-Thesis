#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <serializable.h>

#include "pixel.h"
#include "resourcepool.h"
#include "texture.hpp"

class Mesh;

class MeshProvider
{
 public:
    virtual ~MeshProvider();

    class ModelResource final : public ISerializable
    {
     public:
        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;

        std::string file;
        float size;
        std::string axisOrder;
        std::vector<std::string> excludeMeshes;
        // TODO
        // std::unordered_map<std::string, > materialOverrides;
        std::unordered_map<std::string, std::vector<std::string>> meshSlots;
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
