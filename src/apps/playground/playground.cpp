#include <iostream>
#include <memory>

#include <CLI/CLI.hpp>

#include <animchar.h>
#include <engine.h>
#include <loadmesh.h>

using namespace std;

static void load_mesh(Engine& engine, const std::string& file, const glm::vec3& pos, float size,
                      const glm::vec3& color, bool flipYZ = false) {
    MeshProvider::ResourceDescriptor desc(file, flipYZ);
    Animchar::MeshRes mesh(engine.getResMgr()->getMeshProvider(), std::move(desc));
    std::unique_ptr<Animchar> entity = std::make_unique<Animchar>(std::move(mesh));
    entity->setLocalTransform(
      glm::scale(glm::translate(glm::mat4(1.f), pos), glm::vec3(size, size, size)));
    entity->setUpdateFunctor([](Entity* entity, const Entity::UpdateData& data) {
        entity->setLocalTransform(
          glm::rotate(entity->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
    });
    engine.getEntityManager()->addEntity(std::move(entity));
}

int main(int argc, char* argv[]) {
    try {
        CLI::App app{"Playground"};
        app.set_help_all_flag("--help-all", "Show all help");

        // std::vector<std::string> files;
        // app.add_option("-f,--files,files", files, "Files or folders to open");
        std::string config = "";
        app.add_option("-c,--config", config, "Path to the engine config file");
        // std::string root = "";
        // app.add_option("-r,--root", root, "Working directory for the specified folder");

        CLI11_PARSE(app, argc, argv)

        if (config == "") {
            std::cout << "No config file given. Exiting..." << std::endl;
            return 1;
        }

        Engine engine(config);
        engine.getRenderer()->getCamera().setLookAt(glm::vec3{0, 1, 0});
        engine.getRenderer()->getCamera().setEyePos(glm::vec3{0, 3, -5});

        load_mesh(engine, "../data/models/Philodendron/philodendron.obj", glm::vec3(2, 0.5, 2), 0.5,
                  glm::vec3());
        load_mesh(engine, "../data/models/FreeCharacters/4/Model/steve.blend",
                  glm::vec3(-2, 0.5, 2), 0.3, glm::vec3(), true);
        load_mesh(engine, "../data/models/FreeCharacters/4/Model/creeper.blend",
                  glm::vec3(-4, 0.5, 2), 0.5, glm::vec3(), true);
        load_mesh(engine, "../data/models/suzanne.obj", glm::vec3(0, 0.5, 0), 0.5,
                  glm::vec3(0, 0, 1));
        load_mesh(engine, "../data/models/Scotty.blend", glm::vec3(0, 0.5, 2), 0.15, glm::vec3(),
                  true);

        MeshProvider::ResourceDescriptor sphereDesc(MeshProvider::ResourceDescriptor::SPHERE);
        Animchar::MeshRes sphereMesh(engine.getResMgr()->getMeshProvider(), std::move(sphereDesc));
        std::unique_ptr<Animchar> sphere = std::make_unique<Animchar>(std::move(sphereMesh));
        sphere->setLocalTransform(glm::scale(glm::translate(glm::mat4(1.f), glm::vec3(2, 0.5, 0)),
                                             glm::vec3(0.5, 0.5, 0.5)));
        sphere->setUpdateFunctor([](Entity* sphere, const Entity::UpdateData& data) {
            sphere->setLocalTransform(
              glm::rotate(sphere->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(sphere));

        MeshProvider::ResourceDescriptor cubeDesc(MeshProvider::ResourceDescriptor::CUBE);
        Animchar::MeshRes cubeMesh(engine.getResMgr()->getMeshProvider(), std::move(cubeDesc));
        std::unique_ptr<Animchar> cube = std::make_unique<Animchar>(std::move(cubeMesh));
        cube->setLocalTransform(glm::scale(glm::translate(glm::mat4(1.f), glm::vec3(-2, 0.5, 0)),
                                           glm::vec3(0.5, 0.5, 0.5)));
        cube->setUpdateFunctor([](Entity* cube, const Entity::UpdateData& data) {
            cube->setLocalTransform(
              glm::rotate(cube->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(cube));

        MeshProvider::ResourceDescriptor groundDesc(MeshProvider::ResourceDescriptor::PLANE);
        Animchar::MeshRes groundMesh(engine.getResMgr()->getMeshProvider(), std::move(groundDesc));
        std::unique_ptr<Animchar> ground = std::make_unique<Animchar>(std::move(groundMesh));
        ground->setLocalTransform(glm::scale(glm::mat4(1.f), glm::vec3(100, 100, 100)));
        engine.getEntityManager()->addEntity(std::move(ground));

        engine.gameLoop();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
