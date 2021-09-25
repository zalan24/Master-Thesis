#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class btBroadphaseInterface;
class btDefaultCollisionConfiguration;
class btCollisionDispatcher;
class btSequentialImpulseConstraintSolver;
class btDiscreteDynamicsWorld;
class btRigidBody;
class btSphereShape;
class btBoxShape;
class btStaticPlaneShape;

using RigidBodyPtr = btRigidBody*;

class Physics
{
 public:
    Physics();
    ~Physics();

    enum Shape
    {
        SPHERE,
        CUBE,
        QUAD
    };

    RigidBodyPtr addRigidBody(Shape shape, float mass, float size, const glm::vec3& pos,
                              const glm::quat& rotation);
    void removeRigidBody(RigidBodyPtr bodyPtr);

    void stepSimulation(float deltaTimeS, int maxSubSteps, float fixedTimeStepS);

    struct RigidBodyState
    {
        glm::vec3 position;
        glm::quat rotation;
    };

    RigidBodyState getRigidBodyState(RigidBodyPtr bodyPtr) const;

 private:
    std::unique_ptr<btBroadphaseInterface> broadphase;
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration;
    std::unique_ptr<btCollisionDispatcher> dispatcher;
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver;
    std::unique_ptr<btDiscreteDynamicsWorld> world;

    std::unordered_map<uint32_t, std::unique_ptr<btSphereShape>> sphereShapes;
    std::unordered_map<uint32_t, std::unique_ptr<btBoxShape>> boxShapes;

    mutable std::mutex mutex;

    static uint32_t encode_size(float size) { return uint32_t(size * float(1 << 16)); }
};
