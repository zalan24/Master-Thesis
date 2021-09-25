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

    struct SizeData
    {
        glm::vec3 size;
        SizeData(const glm::vec3& _size) : size(_size) {}
        bool operator<(const SizeData& s) const {
            if (size.x != s.size.x)
                return size.x < s.size.x;
            if (size.y != s.size.y)
                return size.y < s.size.y;
            return size.z < s.size.z;
        }
    };

    std::map<SizeData, std::unique_ptr<btSphereShape>> sphereShapes;
    std::map<SizeData, std::unique_ptr<btBoxShape>> boxShapes;

    mutable std::mutex mutex;
};
