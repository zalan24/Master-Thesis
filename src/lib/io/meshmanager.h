#pragma once

#include <cassert>
#include <memory>
#include <string>

class Scene;
class Mesh;
class SphereMesh;
class Sampler;

class MeshManager
{
 public:
    static MeshManager* getSingleton() {
        assert(instance != nullptr);
        return instance;
    }

    MeshManager();
    ~MeshManager();

    void setScene(Scene* scene);

    void load(const std::string& path);
    void load(std::unique_ptr<Mesh>&& mesh, const std::string& name);
    void save() const;

    struct ModelData
    {
        std::string name = "";
        Mesh* mesh = nullptr;
        SphereMesh* sphereMesh = nullptr;
        Sampler* sampler = nullptr;
    };

    ModelData getCurrentModel() const { return current; }

 private:
    static MeshManager* instance;

    Scene* scene = nullptr;

    ModelData current;
};
