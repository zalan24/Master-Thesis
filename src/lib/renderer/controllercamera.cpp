#include "controllercamera.h"

ControllerCamera::ControllerCamera(Renderer* renderer)
  : CameraHolderEntity(renderer, Entity::AffineTransform(1.f)) {
}
