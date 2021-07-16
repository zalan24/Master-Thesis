#pragma once

#include <functional>
#include <limits>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <serializable.h>

struct Entity final : public IAutoSerializable<Entity>
{
    using EntityId = uint32_t;
    static constexpr EntityId INVALID_ENTITY = std::numeric_limits<EntityId>::max();

    Entity()
      : name(""),
        parent(INVALID_ENTITY),
        position(glm::vec2(0, 0)),
        scale(glm::vec2(1, 1)),
        speed(glm::vec2(0, 0)),
        textureName(""),
        mass(0),
        engineBehaviour(0),
        gameBehaviour(0),
        zPos(0) {}

    REFLECTABLE(
        (std::string) name,
        (EntityId) parent,
        (glm::vec2) position,
        (glm::vec2) scale,
        (glm::vec2) speed,
        (std::string) textureName,
        (float) mass,
        (uint64_t) engineBehaviour,
        (uint64_t) gameBehaviour,
        (float) zPos
    );
};
