#pragma once

#include "cameraholderentity.h"

class Renderer;

class ControllerCamera final : public CameraHolderEntity
{
 public:
    ControllerCamera(Renderer* renderer);

 private:
};
