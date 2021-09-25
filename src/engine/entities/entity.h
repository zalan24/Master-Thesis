#pragma once

#include <functional>
#include <limits>
#include <shared_mutex>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <serializable.h>

struct Entity final : public IAutoSerializable<Entity>
{
    using EntityId = int32_t;
    static constexpr EntityId INVALID_ENTITY = -1;

    Entity()
      : name(""),
        templateName(""),
        parentName(""),
        albedo(glm::vec3(1, 1, 1)),
        position(glm::vec3(0, 0, 0)),
        scale(glm::vec3(1, 1, 1)),
        rotation(),
        modelName(""),
        mass(0),
        hidden(false) {}

    REFLECTABLE((std::string)name, (std::string)templateName, (std::string)parentName,
                (glm::vec3)albedo, (glm::vec3)position, (glm::vec3)velocity, (glm::vec3)scale,
                (glm::quat)rotation, (std::string)modelName, (float)mass, (bool)hidden,
                (std::unordered_map<std::string, float>)extra)

    uint64_t engineBehaviour = 0;
    uint64_t gameBehaviour = 0;
    // uint32_t textureId = 0;
    EntityId parent = INVALID_ENTITY;
    void* rigidBody = nullptr;
    mutable std::shared_mutex mutex;

    Entity(const Entity& other)
      : name(other.name),
        templateName(other.templateName),
        parentName(other.parentName),
        albedo(other.albedo),
        position(other.position),
        velocity(other.velocity),
        scale(other.scale),
        rotation(other.rotation),
        modelName(other.modelName),
        mass(other.mass),
        hidden(other.hidden),
        extra(other.extra),
        engineBehaviour(other.engineBehaviour),
        gameBehaviour(other.gameBehaviour),
        // textureId(other.textureId),
        parent(other.parent),
        rigidBody(other.rigidBody) {}
    Entity& operator=(const Entity& other) {
        if (this == &other)
            return *this;
        name = other.name;
        templateName = other.templateName;
        parentName = other.parentName;
        albedo = other.albedo;
        position = other.position;
        velocity = other.velocity;
        scale = other.scale;
        rotation = other.rotation;
        modelName = other.modelName;
        mass = other.mass;
        hidden = other.hidden;
        extra = other.extra;
        engineBehaviour = other.engineBehaviour;
        gameBehaviour = other.gameBehaviour;
        // textureId = other.textureId;
        parent = other.parent;
        rigidBody = other.rigidBody;
        return *this;
    }
    Entity(Entity&& other)
      : name(std::move(other.name)),
        templateName(std::move(other.templateName)),
        parentName(other.parentName),
        albedo(other.albedo),
        position(other.position),
        velocity(other.velocity),
        scale(other.scale),
        rotation(other.rotation),
        modelName(std::move(other.modelName)),
        mass(other.mass),
        hidden(other.hidden),
        extra(other.extra),
        engineBehaviour(other.engineBehaviour),
        gameBehaviour(other.gameBehaviour),
        // textureId(other.textureId),
        parent(other.parent),
        rigidBody(other.rigidBody) {
        other.engineBehaviour = 0;
        other.gameBehaviour = 0;
    }
    Entity& operator=(Entity&& other) {
        if (this == &other)
            return *this;
        name = std::move(other.name);
        templateName = std::move(other.templateName);
        parentName = other.parentName;
        albedo = other.albedo;
        position = other.position;
        velocity = other.velocity;
        scale = other.scale;
        rotation = other.rotation;
        modelName = std::move(other.modelName);
        mass = other.mass;
        hidden = other.hidden;
        extra = other.extra;
        engineBehaviour = other.engineBehaviour;
        gameBehaviour = other.gameBehaviour;
        // textureId = other.textureId;
        parent = other.parent;
        rigidBody = other.rigidBody;
        other.engineBehaviour = 0;
        other.gameBehaviour = 0;
        return *this;
    }
};
