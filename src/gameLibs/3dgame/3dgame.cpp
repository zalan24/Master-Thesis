#include "3dgame.h"

Game3D::Game3D(int argc, char* argv[], const EngineConfig& config,
               const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
               const Resources& _resources, const Args& args)
  : Engine(argc, argv, config, trackingConfig, shaderbinFile, _resources, args) {
}

Game3D::~Game3D() {
}
