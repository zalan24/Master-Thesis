#include <iostream>
#include <memory>

#include <CLI/CLI.hpp>

#include <animchar.h>
#include <engine.h>

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

        std::unique_ptr<Animchar> ground(new Animchar());
        engine.getEntityManager()->addEntity(std::move(ground));

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
