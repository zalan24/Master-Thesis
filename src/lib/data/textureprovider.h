#pragma once

#include <string>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <serializable.h>

#include "pixel.h"
#include "resourcepool.h"
#include "texture.hpp"

class TextureProvider
{
 public:
    virtual ~TextureProvider() {}

    class ResourceDescriptor final : public ISerializable
    {
     public:
        ResourceDescriptor(const glm::vec4& value);
        ResourceDescriptor(const std::string& filename);

        bool isFile() const { return filename != ""; }
        bool isColor() const { return !isFile(); }

        glm::vec4 getColor() const { return value; }
        const std::string& getFilename() const { return filename; }

        bool operator==(const ResourceDescriptor& other) const {
            if (filename != other.filename)
                return false;
            return filename == "" ? value == other.value : true;
        }

        void writeJson(json& out) const override final;
        void readJson(const json& in) override final;

     private:
        glm::vec4 value;
        std::string filename;
    };

    virtual GenericResourcePool::ResourceRef getResource(const ResourceDescriptor& desc) const = 0;

    virtual GenericResourcePool::ResourceRef createResource(const Texture<RGBA>& tex) const = 0;
    virtual GenericResourcePool::ResourceRef createResource(const RGBA& color) const;
    virtual GenericResourcePool::ResourceRef createResource(const std::string& filename) const = 0;

 private:
};

namespace std
{
template <>
struct hash<TextureProvider::ResourceDescriptor>
{
    std::size_t operator()(TextureProvider::ResourceDescriptor const& s) const noexcept {
        if (s.isFile())
            return std::hash<std::string>{}(s.getFilename());
        if (s.isColor()) {
            glm::vec4 color = s.getColor();
            std::size_t r = std::hash<float>{}(color.r);
            std::size_t g = std::hash<float>{}(color.g);
            std::size_t b = std::hash<float>{}(color.b);
            std::size_t a = std::hash<float>{}(color.a);
            return r ^ g ^ b ^ a;
        }
        assert(false);
        return 0;
    }
};
}  // namespace std
