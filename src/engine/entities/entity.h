#pragma once

#include <functional>
#include <limits>
#include <shared_mutex>
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
        templateName(""),
        parent(INVALID_ENTITY),
        position(glm::vec2(0, 0)),
        scale(glm::vec2(1, 1)),
        speed(glm::vec2(0, 0)),
        textureName(""),
        mass(0),
        zPos(0) {}

    REFLECTABLE(
        (std::string) name,
        (std::string) templateName,
        (EntityId) parent,
        (glm::vec2) position,
        (glm::vec2) scale,
        (glm::vec2) speed,
        (std::string) textureName,
        (float) mass,
        (float) zPos
    )

    uint64_t engineBehaviour = 0;
    uint64_t gameBehaviour = 0;
    uint32_t textureId = 0;
    mutable std::shared_mutex mutex;

    Entity(const Entity& other)
      : name(other.name),
        templateName(other.templateName),
        parent(other.parent),
        position(other.position),
        scale(other.scale),
        speed(other.speed),
        textureName(other.textureName),
        mass(other.mass),
        zPos(other.zPos),
        engineBehaviour(other.engineBehaviour),
        gameBehaviour(other.gameBehaviour) {}
    Entity& operator=(const Entity& other) {
        if (this == &other)
            return *this;
        name = other.name;
        templateName = other.templateName;
        parent = other.parent;
        position = other.position;
        scale = other.scale;
        speed = other.speed;
        textureName = other.textureName;
        mass = other.mass;
        zPos = other.zPos;
        engineBehaviour = other.engineBehaviour;
        gameBehaviour = other.gameBehaviour;
        return *this;
    }
    Entity(Entity&& other)
      : name(std::move(other.name)),
        templateName(std::move(other.templateName)),
        parent(other.parent),
        position(other.position),
        scale(other.scale),
        speed(other.speed),
        textureName(std::move(other.textureName)),
        mass(other.mass),
        zPos(other.zPos),
        engineBehaviour(other.engineBehaviour),
        gameBehaviour(other.gameBehaviour) {
        other.engineBehaviour = 0;
        other.gameBehaviour = 0;
    }
    Entity& operator=(Entity&& other) {
        if (this == &other)
            return *this;
        name = std::move(other.name);
        templateName = std::move(other.templateName);
        parent = other.parent;
        position = other.position;
        scale = other.scale;
        speed = other.speed;
        textureName = std::move(other.textureName);
        mass = other.mass;
        zPos = other.zPos;
        engineBehaviour = other.engineBehaviour;
        gameBehaviour = other.gameBehaviour;
        other.engineBehaviour = 0;
        other.gameBehaviour = 0;
        return *this;
    }
};
