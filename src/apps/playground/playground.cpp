#include <iostream>
#include <memory>

#include <CLI/CLI.hpp>

#include <animchar.h>
#include <engine.h>
#include <loadmesh.h>

using namespace std;

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
        engine.getRenderer()->getCamera().setLookAt(glm::vec3{0, 0, 0});
        engine.getRenderer()->getCamera().setEyePos(glm::vec3{0, 3, -5});

        std::unique_ptr<Animchar> cube = std::make_unique<Animchar>(
          createCube(0.5), nullptr, glm::translate(glm::mat4(1.f), glm::vec3(-2, 0.5, 0)));
        cube->setUpdateFunctor([](Entity* cube, const Entity::UpdateData& data) {
            cube->setLocalTransform(
              glm::rotate(cube->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(cube));
        std::unique_ptr<Animchar> sphere = std::make_unique<Animchar>(
          createSphere(64, 32, 0.5), nullptr, glm::translate(glm::mat4(1.f), glm::vec3(2, 0.5, 0)));
        sphere->setUpdateFunctor([](Entity* sphere, const Entity::UpdateData& data) {
            sphere->setLocalTransform(
              glm::rotate(sphere->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(sphere));

        Mesh suzanneMesh = loadMesh("../data/models/suzanne.obj");
        std::unique_ptr<Entity> suzanne = std::make_unique<Animchar>(
          suzanneMesh, nullptr,
          glm::scale(glm::translate(glm::mat4(1.f), glm::vec3(0, 0.5, 0)),
                     glm::vec3(0.5f, 0.5f, 0.5f)));
        suzanne->setUpdateFunctor([](Entity* suzanne, const Entity::UpdateData& data) {
            suzanne->setLocalTransform(
              glm::rotate(suzanne->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(suzanne));

        Mesh HammerMesh = loadMesh("../data/models/Hammer/Hammer.obj");
        std::unique_ptr<Animchar> Hammer = std::make_unique<Animchar>(
          HammerMesh, nullptr,
          glm::scale(glm::translate(glm::mat4(1.f), glm::vec3(-2, 0.5, 2)),
                     glm::vec3(0.5f, 0.5f, 0.5f)));
        Hammer->setUpdateFunctor([](Entity* Hammer, const Entity::UpdateData& data) {
            Hammer->setLocalTransform(
              glm::rotate(Hammer->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(Hammer));

        Mesh scottyMesh = loadMesh("../data/models/Scotty.blend");
        glm::mat4 scottyTm = scottyMesh.getNodeTm();
        std::swap(scottyTm[1], scottyTm[2]);
        scottyTm = glm::scale(scottyTm, glm::vec3{0.3, 0.3, 0.3});
        scottyMesh.setNodeTm(scottyTm);
        std::unique_ptr<Animchar> scotty = std::make_unique<Animchar>(
          scottyMesh, nullptr,
          glm::scale(glm::translate(glm::mat4(1.f), glm::vec3(0, 0.5, 2)),
                     glm::vec3(0.5f, 0.5f, 0.5f)));
        scotty->setUpdateFunctor([](Entity* scotty, const Entity::UpdateData& data) {
            scotty->setLocalTransform(
              glm::rotate(scotty->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(scotty));

        engine.getEntityManager()->addEntity(
          std::make_unique<Animchar>(createPlane(glm::vec3{0, 0, 0}, glm::vec3(0, 1, 0), 10)));

        engine.gameLoop();
        // FileManager fileManager{argc == 1 ? "" : std::string{argv[1]}};
        // MeshManager meshManager;
        // Window window{1600, 900, "Animations"};
        // std::unique_ptr<Scene> scene = std::make_unique<Scene>();
        // meshManager.setScene(scene.get());
        // window.getRenderer()->setScene(std::move(scene));
        // window.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
