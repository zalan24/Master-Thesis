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
