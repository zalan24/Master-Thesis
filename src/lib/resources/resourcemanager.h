#pragma once

#include <resourcepool.hpp>

class GlTexture;
class GlMesh;
// class Mesh;

// mesh {cpu data; &material}
// material {&texture}
// texture {cpu data}
// gltexture {gpu data}

// rendering: {mesh gpu data; material; gltexture}

class ResourceManager
{
 public:
    using GlTextureRef = ResourcePool<GlTexture>::ResourceRef;
    using GlMeshRef = ResourcePool<GlMesh>::ResourceRef;

    static ResourceManager* getSingleton() { return instance; }

    ResourceManager();
    ~ResourceManager();

    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;

    ResourcePool<GlMesh>* getGlMeshPool() { return &glMeshPool; }

 private:
    static ResourceManager* instance;

    ResourcePool<GlMesh> glMeshPool;
    ResourcePool<GlTexture> glTexturePool;
};
