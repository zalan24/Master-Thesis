#include "input.h"

using namespace drv;

Input::Input(size_t inputBufferSize) : eventQueue(inputBufferSize) {
}

void Input::pushMouseMove(MouseMoveEvent&& event) {
    InputEvent e;
    e.type = InputEvent::MOUSE_MOVE;
    e.event.mouseMove = std::move(event);
    eventQueue.try_enqueue(std::move(e));
}

void Input::pushMouseButton(MouseButtenEvent&& event) {
    InputEvent e;
    e.type = InputEvent::MOUSE_BUTTON;
    e.event.mouseButton = std::move(event);
    eventQueue.try_enqueue(std::move(e));
}

void Input::pushKeyboard(KeyboardEvent&& event) {
    InputEvent e;
    e.type = InputEvent::KEYBOARD;
    e.event.keyboard = std::move(event);
    eventQueue.try_enqueue(std::move(e));
}

void Input::pushScroll(ScrollEvent&& event) {
    InputEvent e;
    e.type = InputEvent::SCROLL;
    e.event.scroll = std::move(event);
    eventQueue.try_enqueue(std::move(e));
}

bool Input::popEvent(InputEvent& event) {
    return eventQueue.try_dequeue(event);
}
