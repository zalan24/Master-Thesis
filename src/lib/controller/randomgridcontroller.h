#pragma once

#include <random>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <serializable.h>

#include "charactercontroller.h"
#include "controllerholder.h"

class RandomGridController : public ISerializableController
{
 public:
    ~RandomGridController() override {}

    ControlData getControls(const IControllable* controlled) override;
    ControlData getControls(const IControllable* controlled) const override;
    void writeJson(json& out) const override final;
    void readJson(const json& in) override final;

    void randomize(bool resetSelection);

 private:
    float maxSpeed = 2;
    float speedDistMul = 10;
    float goalDistance = 0.5;
    glm::mat4 worldTransform;
    glm::mat4 controlProjection;  // meant to map (x,y,z) -> (x,z)
    int resolutionX = -1;
    int resolutionY = -1;
    int seed = -1;

    glm::ivec2 movementSelection;
    glm::ivec2 lookSelection;
    glm::ivec2 faceSelection;

    std::random_device r;
    std::default_random_engine randomEngine;

    void reset();
    glm::vec3 getSelectionPos(glm::ivec2 coord) const;
};
