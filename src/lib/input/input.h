#pragma once

#include <concurrentqueue.h>

class Input
{
 public:
    struct MouseMoveEvent
    {
        double relX;
        double relY;
    };
    struct MouseButtenEvent
    {
        enum Type
        {
            PRESS,
            RELEASE
        } type;
        int buttonId;
        // GLFW has modifier keys here. They could be added if necessary
    };
    struct ScrollEvent
    {
        double x;
        double y;
    };
    struct KeyboardEvent
    {
        enum Type
        {
            PRESS,
            RELEASE,
            REPEAT
        } type;
        int key;
        int scancode;
        // GLFW has modifier keys here. They could be added if necessary
    };
    struct InputEvent
    {
        enum Type
        {
            MOUSE_MOVE,
            MOUSE_BUTTON,
            KEYBOARD,
            SCROLL
        } type;
        union Event
        {
            MouseMoveEvent mouseMove;
            MouseButtenEvent mouseButton;
            KeyboardEvent keyboard;
            ScrollEvent scroll;
        } event;
    };

    Input(size_t inputBufferSize);

    void pushMouseMove(MouseMoveEvent&& event);
    void pushMouseButton(MouseButtenEvent&& event);
    void pushKeyboard(KeyboardEvent&& event);
    void pushScroll(ScrollEvent&& event);

    bool popEvent(InputEvent& event);

 private:
    moodycamel::ConcurrentQueue<InputEvent> eventQueue;
};
