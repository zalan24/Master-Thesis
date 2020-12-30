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
        engine.getRenderer()->getCamera().setEyePos(glm::vec3{0, 2, -3});

        std::unique_ptr<Animchar> cube = std::make_unique<Animchar>(
          createCube(0.5), nullptr, glm::translate(glm::mat4(1.f), glm::vec3(-1, 0.5, 0)));
        cube->setUpdateFunctor([](Entity* cube, const Entity::UpdateData& data) {
            cube->setLocalTransform(
              glm::rotate(cube->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(cube));
        std::unique_ptr<Animchar> sphere = std::make_unique<Animchar>(
          createSphere(64, 32, 0.5), nullptr, glm::translate(glm::mat4(1.f), glm::vec3(1, 0.5, 0)));
        sphere->setUpdateFunctor([](Entity* sphere, const Entity::UpdateData& data) {
            sphere->setLocalTransform(
              glm::rotate(sphere->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
        });
        engine.getEntityManager()->addEntity(std::move(sphere));
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
