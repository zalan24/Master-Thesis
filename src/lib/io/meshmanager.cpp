#include "meshmanager.h"

#include "filemanager.h"
#include "loadmesh.h"
#include "mesh.h"
#include "sampler.h"
#include "scene.h"
#include "spheremesh.h"

MeshManager* MeshManager::instance = nullptr;

MeshManager::MeshManager() {
    assert(instance == nullptr);
    instance = this;
}

MeshManager::~MeshManager() {
    instance = this;
}

void MeshManager::setScene(Scene* s) {
    scene = s;
}

void MeshManager::load(const std::string& path) {
    std::unique_ptr<Mesh> mesh;
    std::vector<std::unique_ptr<Mesh>> meshes = loadMesh(path);
    if (meshes.size() != 1)
        throw std::runtime_error("File (" + path + ") contains " + std::to_string(meshes.size()));
    // cout << "Number of meshes loaded: " << meshes.size() << endl;
    mesh = std::move(meshes[0]);
    load(std::move(mesh), path);
}

void MeshManager::load(std::unique_ptr<Mesh>&& mesh, const std::string& name) {
    mesh->normalize();
    scene->clear();

    std::unique_ptr<SphereMesh> spheremesh;
    std::ifstream in = FileManager::getSingleton()->readCache("spheremesh_" + name);
    if (in.is_open()) {
        spheremesh.reset(new SphereMesh{});
        spheremesh->read(in);
        std::cout << "Loading spheremesh from cache" << std::endl;
    }
    else {
        spheremesh.reset(new SphereMesh{*mesh});
    }

    std::unique_ptr<Sampler> sampler{new Sampler{}};
    std::ifstream inSampler = FileManager::getSingleton()->readCache("sampler_" + name);
    if (inSampler.is_open()) {
        sampler->read(inSampler);
        std::cout << "Loading sampler from cache" << std::endl;
    }

    current.mesh = mesh.get();
    current.sphereMesh = spheremesh.get();
    current.name = name;
    current.sampler = sampler.get();

    mesh->hide();
    mesh->setName("Mesh");
    spheremesh->hide();
    spheremesh->setName("Sphere Mesh");
    sampler->setName("Sampler");
    scene->addItem(Scene::ItemData{std::move(mesh)});
    scene->addItem(Scene::ItemData{std::move(spheremesh)});
    scene->addItem(Scene::ItemData{std::move(sampler)});
}

void MeshManager::save() const {
    if (current.name == "" || current.sphereMesh == nullptr)
        return;

    auto saveObj = [](const ISerializable* obj, const std::string& name) {
        std::ofstream out = FileManager::getSingleton()->writeCache(name);
        if (!out.is_open())
            throw std::runtime_error("Could not save: " + name);
        obj->write(out);
    };

    saveObj(current.sphereMesh, "spheremesh_" + current.name);
    saveObj(current.sampler, "sampler_" + current.name);
}
