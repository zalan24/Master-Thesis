#pragma once

#include <cstdint>

#include <serializable.h>
struct CompileOptions final : public ISerializable
{
    void writeJson(json& out) const override;
    void readJson(const json& in) override;
};
