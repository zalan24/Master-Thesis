#include <iostream>
#include <memory>

#include <CLI/CLI.hpp>

// #include <animchar.h>
#include <engine.h>
#include <loadmesh.h>

using namespace std;

// static Animchar* load_mesh(Engine& engine, const std::string& file, const glm::vec3& pos,
//                            float size, const glm::vec3& color, bool flipYZ = false) {
//     MeshProvider::ResourceDescriptor desc(file, flipYZ);
//     Animchar::MeshRes mesh(engine.getResMgr()->getMeshProvider(), std::move(desc));
//     std::unique_ptr<Animchar> entity = std::make_unique<Animchar>(std::move(mesh));
//     entity->setLocalTransform(
//       glm::scale(glm::translate(glm::mat4(1.f), pos), glm::vec3(size, size, size)));
//     entity->setUpdateFunctor([](Entity* entity, const Entity::UpdateData& data) {
//         entity->setLocalTransform(
//           glm::rotate(entity->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
//     });
//     Animchar* ret = entity.get();
//     engine.getEntityManager()->addEntity(std::move(entity));
//     return ret;
// }

// static shared_ptr<Material> getMat(Engine& engine, TextureProvider::ResourceDescriptor&& desc) {
//     Material::DiffuseRes diffuseRes(engine.getResMgr()->getTexProvider(), std::move(desc));
//     return std::make_unique<Material>(std::move(diffuseRes));
// }

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

        engine.gameLoop();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
