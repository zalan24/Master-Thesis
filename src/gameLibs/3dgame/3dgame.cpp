#include "3dgame.h"

Game3D::Game3D(int argc, char* argv[], const EngineConfig& config,
               const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
               const Args& args)
  : Engine(argc, argv, config, trackingConfig, shaderbinFile, args) {
}

Game3D::~Game3D() {
}
