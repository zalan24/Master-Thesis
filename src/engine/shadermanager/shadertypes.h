#pragma once

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

using mat44 = glm::mat4;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using int2 = glm::ivec2;
using int3 = glm::ivec3;
using int4 = glm::ivec4;

static constexpr float float_default_value = 0.f;
static constexpr int int_default_value = 0;
static constexpr vec2 vec2_default_value = vec2{0, 0};
static constexpr vec3 vec3_default_value = vec3{0, 0, 0};
static constexpr vec4 vec4_default_value = vec4{0, 0, 0, 0};
static constexpr int2 int2_default_value = int2{0, 0};
static constexpr int3 int3_default_value = int3{0, 0, 0};
static constexpr int4 int4_default_value = int4{0, 0, 0, 0};
static constexpr mat44 mat44_default_value = mat44(1.f);
