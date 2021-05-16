#include "3dgame.h"

Game3D::Game3D(int argc, char* argv[], const Config& config,
               const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
               ResourceManager::ResourceInfos resource_infos, const Args& args)
  : Engine(argc, argv, config, trackingConfig, shaderbinFile, resource_infos, args) {
}

Game3D::~Game3D() {
}
