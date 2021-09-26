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
                                   const glm::vec3& _pos, const glm::quat& _rotation,
                                   const glm::vec3& _initialVelocity) {
    // new btStaticPlaneShape();
    btCollisionShape* shape = nullptr;
    if (_shape == SPHERE)
        shape = new btSphereShape(btScalar(_size.x));
    else if (_shape == CUBE)
        shape = new btBoxShape(btVector3(_size.x, _size.y, _size.z));
    if (!shape)
        throw std::runtime_error("Shape could not be initialized");

    // left handed -> right handed
    btQuaternion rotation(-_rotation.x, -_rotation.y, _rotation.z, _rotation.w);
    btVector3 position = btVector3(_pos.x, _pos.y, -_pos.z);
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
        body->setLinearVelocity(
          btVector3(_initialVelocity.x, _initialVelocity.y, -_initialVelocity.z));
    }
    return body;
}

void Physics::removeRigidBody(RigidBodyPtr bodyPtr) {
    {
        std::unique_lock<std::mutex> lock(mutex);
        world->removeRigidBody(bodyPtr);
    }
    delete bodyPtr->getMotionState();
    delete bodyPtr->getCollisionShape();
    delete bodyPtr;
}

void Physics::stepSimulation(float deltaTimeS, int maxSubSteps, float fixedTimeStepS) {
    std::unique_lock<std::mutex> lock(mutex);
    world->stepSimulation(deltaTimeS, maxSubSteps, fixedTimeStepS);
}

Physics::RigidBodyState Physics::getRigidBodyState(RigidBodyPtr bodyPtr) const {
    std::unique_lock<std::mutex> lock(mutex);
    btTransform tm = bodyPtr->getWorldTransform();
    btQuaternion rotation = bodyPtr->getOrientation();
    // btQuaternion rotation = tm.getRotation();
    btVector3 position = tm.getOrigin();
    btVector3 velocity = bodyPtr->getLinearVelocity();
    RigidBodyState ret;
    ret.position = glm::vec3(position.getX(), position.getY(), -position.getZ());
    // right handed -> left handed
    // glm::quat(w, x, y, z), because why not
    ret.rotation = glm::quat(rotation.getW(), -rotation.getX(), -rotation.getY(), rotation.getZ());
    ret.velocity = glm::vec3(velocity.getX(), -velocity.getY(), -velocity.getZ());
    return ret;
}
