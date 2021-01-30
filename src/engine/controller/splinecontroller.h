#pragma once

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

#include <serializable.h>
#include <spline.hpp>

#include "charactercontroller.h"
#include "controllerholder.h"

class SplineController : public ISerializableController
{
 public:
    ~SplineController() override {}

    ControlData getControls(const IControllable* controlled) override;
    ControlData getControls(const IControllable* controlled) const override;
    void writeJson(json& out) const override final;
    void readJson(const json& in) override final;

 private:
    using Spline3D = Spline<glm::vec3, float>;
    Spline3D spline;
    float targetDistance;
    float splineSpeed;
    float maxSpeed = 2;
    float speedDistMul = 10;
    float goalDistance = 0.5;
};
