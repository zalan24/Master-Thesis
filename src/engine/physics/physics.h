#pragma once

#include <map>
#include <memory>
#include <mutex>

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
        CUBE
    };

    RigidBodyPtr addRigidBody(Shape shape, float mass, const glm::vec3& size, const glm::vec3& pos,
                              const glm::quat& rotation, const glm::vec3& initialVelocity);
    void removeRigidBody(RigidBodyPtr bodyPtr);

    void stepSimulation(float deltaTimeS, int maxSubSteps, float fixedTimeStepS);

    struct RigidBodyState
    {
        glm::vec3 position;
        glm::quat rotation;
        glm::vec3 velocity;
    };

    RigidBodyState getRigidBodyState(RigidBodyPtr bodyPtr) const;

 private:
    std::unique_ptr<btBroadphaseInterface> broadphase;
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration;
    std::unique_ptr<btCollisionDispatcher> dispatcher;
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver;
    std::unique_ptr<btDiscreteDynamicsWorld> world;

    mutable std::mutex mutex;
};
