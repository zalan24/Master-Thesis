#pragma once

#include <functional>
#include <variant>
#include <vector>

#include "input.h"
#include "inputlistener.h"

namespace drv
{
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

    using RegisterEvent = std::variant<InputListener*, Listener>;

    bool inListenerLoop = false;
    std::vector<RegisterEvent> registerEvent;

    void setCursorMode();
};

}  // namespace drv
