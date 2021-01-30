#include "inputlistener.h"

using namespace drv;

InputListener::~InputListener() {
}

bool InputListener::process(const Input::InputEvent& event) {
    switch (event.type) {
        case Input::InputEvent::KEYBOARD:
            return processKeyboard(event.event.keyboard);
        case Input::InputEvent::MOUSE_BUTTON:
            return processMouseButton(event.event.mouseButton);
        case Input::InputEvent::MOUSE_MOVE:
            return processMouseMove(event.event.mouseMove);
        case Input::InputEvent::SCROLL:
            return processScroll(event.event.scroll);
    }
    return blockAll;
}
