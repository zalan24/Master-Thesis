#include <iostream>
#include <memory>

#include <filemanager.h>
#include <loadmesh.h>
#include <meshmanager.h>
#include <renderer.h>
#include <scene.h>
#include <spheremesh.h>
#include <window.h>

using namespace std;

int main(int argc, char* argv[]) {
    try {
        FileManager fileManager{argc == 1 ? "" : std::string{argv[1]}};
        MeshManager meshManager;
        Window window{1600, 900, "Animations"};
        std::unique_ptr<Scene> scene = std::make_unique<Scene>();
        meshManager.setScene(scene.get());
        window.getRenderer()->setScene(std::move(scene));
        window.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
