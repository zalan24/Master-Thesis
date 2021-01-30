#include "inputmanager.h"

#include <algorithm>
#include <cassert>

using namespace drv;

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
    if (inListenerLoop)
        registerEvent.push_back(Listener{listener, priority});
    else {
        inputListeners.push_back({listener, priority});
        std::sort(inputListeners.begin(), inputListeners.end(), std::greater<>());
        setCursorMode();
    }
}

void InputManager::unregisterListener(InputListener* listener) {
    auto itr = std::find_if(inputListeners.begin(), inputListeners.end(),
                            [listener](const Listener& l) { return l.ptr == listener; });
    if (itr == inputListeners.end())
        return;
    if (inListenerLoop)
        registerEvent.push_back(itr->ptr);
    else {
        inputListeners.erase(itr);
        // This might not be needed (not sure if erase is order preserving)
        std::sort(inputListeners.begin(), inputListeners.end(), std::greater<>());
        setCursorMode();
    }
}

void InputManager::feedInput(Input::InputEvent&& event) {
    try {
        inListenerLoop = true;
        for (Listener& l : inputListeners)
            if (l.ptr->process(event))
                break;
        inListenerLoop = false;
        if (!registerEvent.empty()) {
            for (size_t i = 0; i < registerEvent.size(); ++i) {
                if (std::holds_alternative<Listener>(registerEvent[i]))
                    inputListeners.push_back(std::move(std::get<Listener>(registerEvent[i])));
                else {
                    assert(std::holds_alternative<InputListener*>(registerEvent[i]));
                    InputListener* ptr = std::get<InputListener*>(registerEvent[i]);
                    auto itr = std::find_if(inputListeners.begin(), inputListeners.end(),
                                            [ptr](const Listener& l) { return l.ptr == ptr; });
                    if (itr != inputListeners.end())
                        inputListeners.erase(itr);
                }
            }
            registerEvent.clear();
            std::sort(inputListeners.begin(), inputListeners.end(), std::greater<>());
            setCursorMode();
        }
    }
    catch (...) {
        inListenerLoop = false;
        throw;
    }
}

void InputManager::setCursorMode() {
    InputListener::CursorMode mode = InputListener::DONT_CARE;
    for (Listener& l : inputListeners)
        if ((mode = l.ptr->getCursorMode()) != InputListener::DONT_CARE)
            break;
    if (cursorCallback)
        cursorCallback(mode);
}

void InputManager::setCursorModeCallbock(CursorModeCallback&& callback) {
    cursorCallback = std::move(callback);
    setCursorMode();
}
