#pragma once

#include "input.h"

class InputListener
{
 public:
    explicit InputListener(bool block_all) : blockAll(block_all) {}
    virtual ~InputListener();

    bool process(const Input::InputEvent& event);

    enum CursorMode
    {
        DONT_CARE,
        NORMAL,
        HIDE,
        LOCK
    };

    virtual CursorMode getCursorMode() { return DONT_CARE; }

 protected:
    virtual bool processKeyboard(const Input::KeyboardEvent&) { return blockAll; }
    virtual bool processMouseButton(const Input::MouseButtenEvent&) { return blockAll; }
    virtual bool processMouseMove(const Input::MouseMoveEvent&) { return blockAll; }
    virtual bool processScroll(const Input::ScrollEvent&) { return blockAll; }
    virtual bool processWindowFocus(const Input::WindowFocusEvent&) { return blockAll; }
    virtual bool processCursorEntered(const Input::CursorEnterEvent&) { return blockAll; }
    virtual bool processChar(const Input::CharEvent&) { return blockAll; }
    virtual bool processMonitor(const Input::MonitorEvent&) { return blockAll; }

 private:
    bool blockAll;
};
