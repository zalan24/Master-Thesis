#include "inputlistener.h"

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
        case Input::InputEvent::WINDOW_FOCUS:
            return processWindowFocus(event.event.windowFocus);
        case Input::InputEvent::CURSOR_ENTERED:
            return processCursorEntered(event.event.cursorEntered);
        case Input::InputEvent::CHAR_EVENT:
            return processChar(event.event.charEvent);
        case Input::InputEvent::MONITOR:
            return processMonitor(event.event.monitor);
    }
    return blockAll;
}
