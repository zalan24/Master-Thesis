#pragma once

#include <string>

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

    class ResourceDescriptor final : public ISerializable
    {
     public:
        enum Type
        {
            FILE,
            CUBE,
            SPHERE,
            PLANE
        };

        ResourceDescriptor(const std::string& filename);
        ResourceDescriptor(Type type);

        bool isFile() const { return type == FILE; }
        Type getType() const { return type; }

        const std::string& getFilename() const { return filename; }

        bool operator==(const ResourceDescriptor& other) const {
            return type == other.type && (type != FILE || filename == other.filename);
        }

     protected:
        void gatherEntries(std::vector<Entry>& entries) const override;

     private:
        Type type;
        std::string filename;
    };

    virtual GenericResourcePool::ResourceRef getResource(const ResourceDescriptor& desc) const = 0;
    virtual GenericResourcePool::ResourceRef createResource(const std::string& filename) const = 0;
    virtual GenericResourcePool::ResourceRef createResource(
      const ResourceDescriptor::Type type) const = 0;
    virtual GenericResourcePool::ResourceRef createResource(const Mesh& m) const = 0;

 private:
};

namespace std
{
template <>
struct hash<MeshProvider::ResourceDescriptor>
{
    std::size_t operator()(MeshProvider::ResourceDescriptor const& s) const noexcept {
        if (s.isFile())
            return std::hash<std::string>{}(s.getFilename());
        return std::hash<MeshProvider::ResourceDescriptor::Type>{}(s.getType());
    }
};
}  // namespace std
