#pragma once

#include <engine.h>

class Game3D : public Engine
{
 public:
    Game3D(int argc, char* argv[], const Config& config,
           const drv::StateTrackingConfig& trackingConfig, const std::string& shaderbinFile,
           ResourceManager::ResourceInfos resource_infos, const Args& args);

    // TODO acquire swapchain image here instead of in game
    // don't call game specific rendering code if no swapchain was received

    virtual ~Game3D() override;

 private:
};
