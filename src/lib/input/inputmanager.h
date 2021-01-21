#pragma once

#include <functional>
#include <vector>

#include "input.h"
#include "inputlistener.h"

class InputManager
{
 public:
    static InputManager* getSingleton() { return instance; }

    using CursorModeCallback = std::function<void(InputListener::CursorMode)>;

    InputManager();
    ~InputManager();

    InputManager(const InputManager&) = delete;
    InputManager& operator=(const InputManager&) = delete;

    InputManager(InputManager&& other);
    InputManager& operator=(InputManager&& other);

    void registerListener(InputListener* listener, float priority);
    void unregisterListener(InputListener* listener);

    void feedInput(Input::InputEvent&& event);

    void setCursorModeCallbock(CursorModeCallback&& callback);

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
    CursorModeCallback cursorCallback;

    void setCursorMode();
};
