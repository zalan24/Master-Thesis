#pragma once

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

using mat4 = glm::mat4;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using ivec2 = glm::ivec2;
using ivec3 = glm::ivec3;
using ivec4 = glm::ivec4;
using uint = uint32_t;
using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;
using uvec4 = glm::uvec4;

static constexpr float float_default_value = 0.f;
static constexpr int int_default_value = 0;
static constexpr vec2 vec2_default_value = vec2{0, 0};
static constexpr vec3 vec3_default_value = vec3{0, 0, 0};
static constexpr vec4 vec4_default_value = vec4{0, 0, 0, 0};
static constexpr ivec2 int2_default_value = ivec2{0, 0};
static constexpr ivec3 int3_default_value = ivec3{0, 0, 0};
static constexpr ivec4 int4_default_value = ivec4{0, 0, 0, 0};
static constexpr uint uint_default_value = 0;
static constexpr uvec2 uint2_default_value = uvec2{0, 0};
static constexpr uvec3 uint3_default_value = uvec3{0, 0, 0};
static constexpr uvec4 uint4_default_value = uvec4{0, 0, 0, 0};
static constexpr mat4 mat4_default_value = mat4(1.f);
