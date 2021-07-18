#include <iostream>
#include <memory>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

// #include <animchar.h>
// #include <controllerholder.h>
// #include <engine.h>
#include <game.h>
// #include <loadmesh.h>

using json = nlohmann::json;

using namespace std;

// static Animchar* create_character(Engine& engine, const std::string& resName,
//                                   const glm::vec3& pos) {
//     MeshProvider::ResourceDescriptor desc(resName);
//     Animchar::MeshRes mesh(std::move(desc));
//     std::unique_ptr<Animchar> entity = std::make_unique<Animchar>(std::move(mesh));
//     entity->setLocalTransform(glm::translate(glm::mat4(1.f), pos));
//     Animchar* ret = entity.get();
//     EntityManager::EntityId id = engine.getEntityManager()->addEntity(std::move(entity));
//     engine.getRenderer()->setCharacter(id);
//     return ret;
// }

// static Animchar* load_mesh(Engine& engine, const std::string& resName, const glm::vec3& pos) {
//     MeshProvider::ResourceDescriptor desc(resName);
//     Animchar::MeshRes mesh(std::move(desc));
//     std::unique_ptr<Animchar> entity = std::make_unique<Animchar>(std::move(mesh));
//     entity->setLocalTransform(glm::translate(glm::mat4(1.f), pos));
//     entity->setUpdateFunctor([](Entity* entity, const Entity::UpdateData& data) {
//         entity->setLocalTransform(
//           glm::rotate(entity->getLocalTransform(), data.dt, glm::vec3{0, 1, 0}));
//     });
//     Animchar* ret = entity.get();
//     engine.getEntityManager()->addEntity(std::move(entity));
//     return ret;
// }

// static Animchar* get_controlled_character(Engine& engine, const std::string& resName,
//                                           const glm::vec3& pos, const json& controllerJson) {
//     MeshProvider::ResourceDescriptor desc(resName);
//     Animchar::MeshRes mesh(std::move(desc));
//     std::unique_ptr<Animchar> entity = std::make_unique<Animchar>(std::move(mesh));
//     entity->setLocalTransform(glm::translate(glm::mat4(1.f), pos));
//     Animchar* ret = entity.get();
//     std::unique_ptr<ControllerHolder> controller = std::make_unique<ControllerHolder>();
//     ISerializable::serialize(controllerJson, *controller.get());
//     ret->setController(std::move(controller));
//     engine.getEntityManager()->addEntity(std::move(entity));
//     return ret;
// }

// static shared_ptr<Material> getMat(Engine& engine, TextureProvider::ResourceDescriptor&& desc) {
//     Material::DiffuseRes diffuseRes(std::move(desc));
//     return std::make_unique<Material>(std::move(diffuseRes));
// }

int main(int argc, char* argv[]) {
    try {
        CLI::App app{"Playground"};
        app.set_help_all_flag("--help-all", "Show all help");

        // std::vector<std::string> files;
        // app.add_option("-f,--files,files", files, "Files or folders to open");
        Engine::Resources resources;

        std::string config = "";
        app.add_option("-c,--config", config, "Path to the engine config file");
        std::string trackingConfig = "";
        app.add_option("-t,--trackingConfig", trackingConfig, "Path to the tracking config file");
        std::string shaderbin = "";
        app.add_option("-s,--shaderbin", shaderbin, "Path to the shader bin file");
        std::string modelResources = "";
        app.add_option("-m,--models", modelResources, "Path to the model resources json file");
        app.add_option("-d,--data", resources.assets, "Path to the data folder");
        Engine::Args args;
        app.add_flag("-r,--renderdoc", args.renderdocEnabled, "Enable renderdoc layer");
        app.add_flag("-g,--gfx", args.gfxCaptureEnabled, "Enable gfx capture layer");
        app.add_flag("--api_dump", args.apiDumpEnabled, "Enable vulkan api dump");
        app.add_option("--scene", args.sceneToLoad, "Which scene file to open");
        app.add_flag("--no_persistance", args.clearRuntimeStats,
                     "Don't load any previously exported runtime stats");
        app.add_option("--runtime_stats_persistance", args.runtimeStatsPersistanceBin,
                       "Binary file for storing runtime statistics persistance");
        app.add_option("--runtime_stats_game_exports", args.runtimeStatsGameExportsBin,
                       "Binary file for storing runtime statistics game exports");
        app.add_option("--runtime_stats_cache", args.runtimeStatsCacheBin,
                       "Binary file for storing runtime statistics cache");
        app.add_option("--report", args.reportFile, "Txt file, where the report will be generated");

        CLI11_PARSE(app, argc, argv)

        resources.textures = (fs::path{resources.assets} / "textures").string();

        if (config == "") {
            std::cout << "No config file given. Exiting..." << std::endl;
            return 1;
        }

        EngineConfig engineCfg;
        {
            if (!engineCfg.importFromFile(fs::path{config}))
                throw std::runtime_error("Could not open engine config file: " + config);
        }
        drv::StateTrackingConfig trackingCfg;
        if (trackingConfig != "") {
            if (!trackingCfg.importFromFile(fs::path{trackingConfig}))
                throw std::runtime_error("Could not open tracking config file: " + trackingConfig);
        }

        // json controllers;
        // {
        //     std::ifstream in(resourceFolder + "/scenes/test_controllers.json");
        //     if (!in.is_open())
        //         throw std::runtime_error("Could not open controllers file");
        //     in >> controllers;
        // }

        // ResourceManager::ResourceInfos resourceInfos;
        // resourceInfos.resourceFolder = resourceFolder;
        // resourceInfos.modelResourcesJson = modelResources;
        std::unique_ptr<Game> game = std::make_unique<Game>(argc, argv, engineCfg, trackingCfg,
                                                            shaderbin, resources, std::move(args));
        // engine.getRenderer()->getCamera().lookAt(glm::vec3{0, 3, -5}, glm::vec3{0, 1, 0},
        //                                          glm::vec3{0, 1, 0});

        // create_character(engine, "BH-2", glm::vec3(0, 0.5, -2));
        // get_controlled_character(engine, "BH-2", glm::vec3(10, 0.5, 0),
        //                          controllers["gridController"]);
        // // load_mesh(engine, "BH-2", glm::vec3(-4, 0.5, 2));
        // // load_mesh(engine, "cartoon_boy", glm::vec3(2, 0.5, 2));
        // // load_mesh(engine, "plant", glm::vec3(2, 0.5, 2));
        // // load_mesh(engine, "boring_guy", glm::vec3(-2, 0.5, 2));
        // load_mesh(engine, "Twilight", glm::vec3(4, 0.5, 2));
        // // load_mesh(engine, "suzanne", glm::vec3(0, 0.5, 0));
        // // load_mesh(engine, "Scotty", glm::vec3(0, 0.5, 2));

        // load_mesh(engine, "ball", glm::vec3(2, 0.5, 0));
        // Animchar* sphere = load_mesh(engine, "ball", glm::vec3(3.5, 0.5, 0));
        // sphere->setMaterial(
        //   getMat(engine, TextureProvider::ResourceDescriptor("textures/earth.jpg")), true);
        // load_mesh(engine, "cube", glm::vec3(-2, 0.5, 0));

        // MeshProvider::ResourceDescriptor groundDesc("ground");
        // Animchar::MeshRes groundMesh(std::move(groundDesc));
        // std::unique_ptr<Animchar> ground = std::make_unique<Animchar>(std::move(groundMesh));
        // ground->setMaterial(
        //   getMat(engine, TextureProvider::ResourceDescriptor(glm::vec4(1, 1, 1, 1))), true);
        // engine.getEntityManager()->addEntity(std::move(ground));

        game->gameLoop();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
