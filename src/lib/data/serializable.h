#pragma once

#include <iostream>

class ISerializable
{
 public:
    virtual ~ISerializable() = 0;

    virtual void write(std::ostream& out) const = 0;
    virtual void read(std::istream& in) = 0;
};
