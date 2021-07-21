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
    using EntityId = int32_t;
    static constexpr EntityId INVALID_ENTITY = -1;

    Entity()
      : name(""),
        templateName(""),
        parentName(""),
        position(glm::vec2(0, 0)),
        scale(glm::vec2(1, 1)),
        speed(glm::vec2(0, 0)),
        textureName(""),
        mass(0),
        zPos(0),
        hidden(false) {}

    REFLECTABLE(
        (std::string) name,
        (std::string) templateName,
        (std::string) parentName,
        (glm::vec2) position,
        (glm::vec2) scale,
        (glm::vec2) speed,
        (std::string) textureName,
        (float) mass,
        (float) zPos,
        (bool) hidden,
        (std::unordered_map<std::string, float>) extra
    )

    uint64_t engineBehaviour = 0;
    uint64_t gameBehaviour = 0;
    uint32_t textureId = 0;
    EntityId parent = INVALID_ENTITY;
    mutable std::shared_mutex mutex;

    Entity(const Entity& other)
      : name(other.name),
        templateName(other.templateName),
        parentName(other.parentName),
        position(other.position),
        scale(other.scale),
        speed(other.speed),
        textureName(other.textureName),
        mass(other.mass),
        zPos(other.zPos),
        hidden(other.hidden),
        extra(other.extra),
        engineBehaviour(other.engineBehaviour),
        gameBehaviour(other.gameBehaviour),
        textureId(other.textureId),
        parent(other.parent) {}
    Entity& operator=(const Entity& other) {
        if (this == &other)
            return *this;
        name = other.name;
        templateName = other.templateName;
        parentName = other.parentName;
        position = other.position;
        scale = other.scale;
        speed = other.speed;
        textureName = other.textureName;
        mass = other.mass;
        zPos = other.zPos;
        hidden = other.hidden;
        extra = other.extra;
        engineBehaviour = other.engineBehaviour;
        gameBehaviour = other.gameBehaviour;
        textureId = other.textureId;
        parent = other.parent;
        return *this;
    }
    Entity(Entity&& other)
      : name(std::move(other.name)),
        templateName(std::move(other.templateName)),
        parentName(other.parentName),
        position(other.position),
        scale(other.scale),
        speed(other.speed),
        textureName(std::move(other.textureName)),
        mass(other.mass),
        zPos(other.zPos),
        hidden(other.hidden),
        extra(other.extra),
        engineBehaviour(other.engineBehaviour),
        gameBehaviour(other.gameBehaviour),
        textureId(other.textureId),
        parent(other.parent) {
        other.engineBehaviour = 0;
        other.gameBehaviour = 0;
    }
    Entity& operator=(Entity&& other) {
        if (this == &other)
            return *this;
        name = std::move(other.name);
        templateName = std::move(other.templateName);
        parentName = other.parentName;
        position = other.position;
        scale = other.scale;
        speed = other.speed;
        textureName = std::move(other.textureName);
        mass = other.mass;
        zPos = other.zPos;
        hidden = other.hidden;
        extra = other.extra;
        engineBehaviour = other.engineBehaviour;
        gameBehaviour = other.gameBehaviour;
        textureId = other.textureId;
        parent = other.parent;
        other.engineBehaviour = 0;
        other.gameBehaviour = 0;
        return *this;
    }
};
