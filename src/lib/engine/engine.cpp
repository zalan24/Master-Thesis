#include "engine.h"

#include <fstream>

void Engine::Config::gatherEntries(std::vector<ISerializable::Entry>& entries) const {
    REGISTER_ENTRY(screenWidth, entries);
    REGISTER_ENTRY(screenHeight, entries);
    REGISTER_ENTRY(title, entries);
}

static Engine::Config get_config(const std::string& file) {
    std::ifstream in(file);
    assert(in.is_open());
    Engine::Config config;
    config.read(in);
    return config;
}

Engine::Engine(const std::string& configFile) : Engine(get_config(configFile)) {
}

Engine::Engine(const Config& cfg)
  : config(cfg), window(config.screenWidth, config.screenHeight, config.title) {
}

Engine::~Engine() {
    // TODO
}

void Engine::gameLoop() {
    // TODO
}
