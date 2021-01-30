#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>

class ShaderManager;

struct RenderContext
{
    int width;
    int height;
    glm::mat4 pv;
    glm::vec3 lightColor;
    glm::vec3 lightDir;
    glm::vec3 ambientColor;
    const ShaderManager* shaderManager;
    unsigned int maxStencil;
    std::atomic<unsigned int>* currentStencil;
    mutable std::mutex mutex;
};

void checkError();
