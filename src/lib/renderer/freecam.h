#pragma once

#include <memory>

#include <inputlistener.h>

#include "cameraholderentity.h"

class FreeCamEntity final : public CameraHolderEntity
{
 public:
    FreeCamEntity(Renderer* renderer);

 protected:
    void _activate() override final;
    void _deactivate() override final;

 private:
    std::unique_ptr<InputListener> inputListener;
};
