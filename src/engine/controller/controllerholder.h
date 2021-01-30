#pragma once

#include <memory>
#include <string>

#include <serializable.h>

#include "charactercontroller.h"

class ISerializableController
  : public ICharacterController
  , public ISerializable
{};

class ControllerHolder final
  : public IVirtualSerializable
  , public ICharacterController
{
 public:
    ISerializable* init(const std::string& type) override final;
    const ISerializable* getCurrent() const override final;
    std::string getCurrentType() const override final;
    void reset() override final;

    ~ControllerHolder() override {}

    ControlData getControls(const IControllable* controlled) const override final;

 private:
    std::string type = "";
    std::unique_ptr<ISerializableController> controller;
};
