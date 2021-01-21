#pragma once

#include <vector>

#include "input.h"
#include "inputlistener.h"

class InputManager
{
 public:
    static InputManager* getSingleton() { return instance; }

    InputManager();
    ~InputManager();

    InputManager(const InputManager&) = delete;
    InputManager& operator=(const InputManager&) = delete;

    InputManager(InputManager&& other);
    InputManager& operator=(InputManager&& other);

    void registerListener(InputListener* listener, float priority);
    void unregisterListener(InputListener* listener);

    void feedInput(Input::InputEvent&& event);

 private:
    static InputManager* instance;

    struct Listener
    {
        InputListener* ptr;
        float priority;
        bool operator<(const Listener& rhs) const { return priority < rhs.priority; }
        bool operator>(const Listener& rhs) const { return priority > rhs.priority; }
    };
    std::vector<Listener> inputListeners;
};
