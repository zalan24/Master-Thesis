#pragma once

#include "input.h"

class InputListener
{
 public:
    explicit InputListener(bool block_all) : blockAll(block_all) {}
    virtual ~InputListener();

    bool process(const Input::InputEvent& event);

 protected:
    virtual bool processKeyboard(const Input::KeyboardEvent&) { return blockAll; }
    virtual bool processMouseButton(const Input::MouseButtenEvent&) { return blockAll; }
    virtual bool processMouseMove(const Input::MouseMoveEvent&) { return blockAll; }
    virtual bool processScroll(const Input::ScrollEvent&) { return blockAll; }

 private:
    bool blockAll;
};
