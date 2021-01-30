#include "randomgridcontroller.h"

glm::vec3 RandomGridController::getSelectionPos(glm::ivec2 coord) const {
    glm::vec3 pos = glm::vec3(float(coord.x) / resolutionX, 0, float(coord.y) / resolutionY) * 2.f
                    - glm::vec3(1.f - 1.f / resolutionX, 0, 1.f - 1.f / resolutionY);
    return worldTransform * glm::vec4(pos, 1);
}

ICharacterController::ControlData RandomGridController::getControls(
  const IControllable* controlled) {
    ICharacterController::ControlData ret =
      const_cast<const RandomGridController*>(this)->getControls(controlled);
    glm::vec3 pos = controlled->getPos();
    glm::vec3 movePos = controlProjection * glm::vec4(getSelectionPos(movementSelection), 1);
    if (glm::length(pos - movePos) < goalDistance)
        randomize(true);
    return ret;
}

ICharacterController::ControlData RandomGridController::getControls(
  const IControllable* controlled) const {
    glm::vec3 pos = controlled->getPos();
    glm::vec3 projPos = controlProjection * glm::vec4(pos, 1);
    glm::vec3 lookPos = getSelectionPos(lookSelection);
    glm::vec3 facePos = controlProjection * glm::vec4(getSelectionPos(faceSelection), 1);
    glm::vec3 movePos = controlProjection * glm::vec4(getSelectionPos(movementSelection), 1);
    ICharacterController::ControlData ret;
    ret.facing.dir = glm::normalize(lookPos - pos);
    ret.looking.dir = glm::normalize(facePos - projPos);
    ret.movement.speed = movePos - pos;
    float speedLen = glm::length(ret.movement.speed);
    if (speedLen > 0)
        ret.movement.speed =
          glm::normalize(ret.movement.speed) * std::min(maxSpeed, speedDistMul * speedLen);
    return ret;
}

void RandomGridController::writeJson(json& out) const {
    WRITE_OBJECT(maxSpeed, out);
    WRITE_OBJECT(speedDistMul, out);
    WRITE_OBJECT(goalDistance, out);
    WRITE_OBJECT(worldTransform, out);
    WRITE_OBJECT(controlProjection, out);
    WRITE_OBJECT(resolutionX, out);
    WRITE_OBJECT(resolutionY, out);
}

void RandomGridController::readJson(const json& in) {
    READ_OBJECT_OPT(maxSpeed, in, 2);
    READ_OBJECT_OPT(speedDistMul, in, 10);
    READ_OBJECT_OPT(goalDistance, in, 0.5);
    READ_OBJECT_OPT(worldTransform, in, glm::mat4(1.f));
    glm::mat4 defProj = glm::mat4(1.f);
    defProj[1][1] = 0;
    READ_OBJECT_OPT(controlProjection, in, defProj);
    READ_OBJECT_OPT(resolutionX, in, 1);
    READ_OBJECT_OPT(resolutionY, in, 1);

    READ_OBJECT_OPT(movementSelection, in, glm::ivec2(-1, -1));
    READ_OBJECT_OPT(lookSelection, in, glm::ivec2(-1, -1));
    READ_OBJECT_OPT(faceSelection, in, glm::ivec2(-1, -1));
    READ_OBJECT_OPT(seed, in, -1);

    if (seed < 0)
        randomEngine = std::default_random_engine(r());
    else
        randomEngine = std::default_random_engine(static_cast<unsigned int>(seed));
    randomize(false);
}

void RandomGridController::reset() {
    movementSelection = glm::ivec2(-1, -1);
    lookSelection = glm::ivec2(-1, -1);
    faceSelection = glm::ivec2(-1, -1);
}

void RandomGridController::randomize(bool resetSelection) {
    if (resolutionX <= 0 || resolutionY <= 0)
        return;
    if (resetSelection)
        reset();
    std::uniform_int_distribution<int> xDist(0, resolutionX - 1);
    std::uniform_int_distribution<int> yDist(0, resolutionY - 1);
    if (movementSelection.x < 0)
        movementSelection.x = xDist(randomEngine);
    if (movementSelection.y < 0)
        movementSelection.y = yDist(randomEngine);
    if (lookSelection.x < 0)
        lookSelection.x = xDist(randomEngine);
    if (lookSelection.y < 0)
        lookSelection.y = yDist(randomEngine);
    if (faceSelection.x < 0)
        faceSelection.x = xDist(randomEngine);
    if (faceSelection.y < 0)
        faceSelection.y = yDist(randomEngine);
}
