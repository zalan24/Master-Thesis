#include "cameraholderentity.h"

#include <entitymanager.h>

#include "renderer.h"

CameraHolderEntity::CameraHolderEntity(Renderer* _renderer,
                                       const Entity::AffineTransform& camera_offset, Entity* parent,
                                       const Entity::AffineTransform& localTm)
  : Entity(parent, localTm), renderer(_renderer), cameraOffset(camera_offset) {
}

void CameraHolderEntity::activate() {
    EntityQuery deactivateQuery(
      [](const Entity* entity) {
          return dynamic_cast<const CameraHolderEntity*>(entity) != nullptr;
      },
      [](Entity* entity) { static_cast<CameraHolderEntity*>(entity)->deactivate(); });
    EntityManager::getSingleton()->performQuery(deactivateQuery);
    _activate();
}

void CameraHolderEntity::deactivate() {
    active = false;
    _deactivate();
}

void CameraHolderEntity::update(const UpdateData& data) {
    _update(data);
    if (active) {
        AffineTransform tm = getWorldTransform() * cameraOffset;
        renderer->getCamera().setLookAt(tm[3], tm[3] + tm[2], tm[1]);
    }
}
