#include "physics.h"

#include <btBulletDynamicsCommon.h>

Physics::Physics()
  : broadphase(std::make_unique<btDbvtBroadphase>()),
    collisionConfiguration(std::make_unique<btDefaultCollisionConfiguration>()),
    dispatcher(std::make_unique<btCollisionDispatcher>(collisionConfiguration.get())),
    solver(std::make_unique<btSequentialImpulseConstraintSolver>()),
    world(std::make_unique<btDiscreteDynamicsWorld>(dispatcher.get(), broadphase.get(),
                                                    solver.get(), collisionConfiguration.get())) {
    world->setGravity(btVector3(0, -9.8f, 0));
}

Physics::~Physics() {
}

RigidBodyPtr Physics::addRigidBody(Shape _shape, float _mass, const glm::vec3& _size,
                                   const glm::vec3& _pos, const glm::quat& _rotation) {
    // new btStaticPlaneShape();
    SizeData sizeCode = SizeData(_size);
    btCollisionShape* shape = nullptr;
    if (_shape == SPHERE) {
        if (auto itr = sphereShapes.find(sizeCode); itr != sphereShapes.end())
            shape = itr->second.get();
        else {
            std::unique_ptr<btSphereShape> sphere =
              std::make_unique<btSphereShape>(btScalar(_size.x));
            shape = sphere.get();
            sphereShapes.insert({sizeCode, std::move(sphere)});
        }
    }
    else if (_shape == CUBE) {
        if (auto itr = boxShapes.find(sizeCode); itr != boxShapes.end())
            shape = itr->second.get();
        else {
            std::unique_ptr<btBoxShape> cube =
              std::make_unique<btBoxShape>(btVector3(_size.x, _size.y, _size.z));
            shape = cube.get();
            boxShapes.insert({sizeCode, std::move(cube)});
        }
    }
    if (!shape)
        throw std::runtime_error("Shape could not be initialized");

    btQuaternion rotation(_rotation.x, _rotation.y, _rotation.z, _rotation.w);
    btVector3 position = btVector3(_pos.x, _pos.y, _pos.z);
    btDefaultMotionState* motionState = new btDefaultMotionState(btTransform(rotation, position));
    btScalar bodyMass = _mass;
    btVector3 bodyInertia;
    shape->calculateLocalInertia(bodyMass, bodyInertia);
    btRigidBody::btRigidBodyConstructionInfo bodyCI =
      btRigidBody::btRigidBodyConstructionInfo(bodyMass, motionState, shape, bodyInertia);
    bodyCI.m_restitution = 0.0f;
    bodyCI.m_friction = 0.5f;
    // bodyCI.

    RigidBodyPtr body = new btRigidBody(bodyCI);

    // body->setUserPointer((__bridge void*)self);

    // can limit movement on axises
    body->setLinearFactor(btVector3(1, 1, 1));

    {
        std::unique_lock<std::mutex> lock(mutex);
        world->addRigidBody(body);
    }
    return body;
}

void Physics::removeRigidBody(RigidBodyPtr bodyPtr) {
    {
        std::unique_lock<std::mutex> lock(mutex);
        world->removeRigidBody(bodyPtr);
    }
    delete bodyPtr->getMotionState();
    delete bodyPtr;
}

void Physics::stepSimulation(float deltaTimeS, int maxSubSteps, float fixedTimeStepS) {
    std::unique_lock<std::mutex> lock(mutex);
    world->stepSimulation(deltaTimeS, maxSubSteps, fixedTimeStepS);
}

Physics::RigidBodyState Physics::getRigidBodyState(RigidBodyPtr bodyPtr) const {
    std::unique_lock<std::mutex> lock(mutex);
    btTransform tm = bodyPtr->getWorldTransform();
    btQuaternion rotation = tm.getRotation();
    btVector3 position = tm.getOrigin();
    RigidBodyState ret;
    ret.position = glm::vec3(position.getX(), position.getY(), position.getZ());
    ret.rotation = glm::quat(rotation.getX(), rotation.getY(), rotation.getZ(), rotation.getW());
    return ret;
}
