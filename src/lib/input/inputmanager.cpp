#include "inputmanager.h"

#include <algorithm>
#include <cassert>

InputManager* InputManager::instance = nullptr;

InputManager::InputManager() {
    assert(instance == nullptr);
    instance = this;
}

InputManager::~InputManager() {
    if (instance == this)
        instance = nullptr;
}

InputManager::InputManager(InputManager&& other) : inputListeners(std::move(other.inputListeners)) {
    if (&other == instance)
        instance = this;
}

InputManager& InputManager::operator=(InputManager&& other) {
    if (&other == this)
        return *this;
    if (&other == instance)
        instance = this;
    inputListeners = std::move(other.inputListeners);
    return *this;
}

void InputManager::registerListener(InputListener* listener, float priority) {
    inputListeners.push_back({listener, priority});
    std::sort(inputListeners.begin(), inputListeners.end(), std::greater<>());
}

void InputManager::unregisterListener(InputListener* listener) {
    auto itr = std::find_if(inputListeners.begin(), inputListeners.end(),
                            [listener](const Listener& l) { return l.ptr == listener; });
    assert(itr != inputListeners.end());
    inputListeners.erase(itr);
    // This might not be needed (not sure if erase is order preserving)
    std::sort(inputListeners.begin(), inputListeners.end(), std::greater<>());
}

void InputManager::feedInput(Input::InputEvent&& event) {
    // TODO deal with unregistered listeners here (might happen in the loop)
    for (Listener& l : inputListeners)
        if (l.ptr->process(event))
            return;
}
