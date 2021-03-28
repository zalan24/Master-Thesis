#pragma once

// This class is meant to be used for resources like image views,
// where an object needs to be created for a parent resource and it's bound to it
// The goal is to handle the case, where the parent object gets recreated

#include "drverror.h"

namespace drv
{
template <typename P, typename T, typename D>
class BoundResource
{
 public:
    using Eq = bool(const D&, const P&, const P&, const T&);  // compare current and new parent
    using Gen = T(const D& const P&);                         // generate new value from parent
    using Del = void(const D&, T&&);                          // delete current value

    BoundResource(D&& _data, Eq&& _eq, Gen&& _gen, Del&& _del)
      : data(std::move(_data)), eq(std::move(_eq)), gen(std::move(_gen)), del(std::move(_del)) {}

    BoundResource(const BoundResource&) = delete;
    BoundResource& operator=(const BoundResource&) = delete;

    BoundResource(BoundResource&& other)
      : parent(std::move(other.parent)),
        value(std::move(other.value)),
        data(std::move(other.data)),
        eq(std::move(other.eq)),
        gen(std::move(other.gen)),
        del(std::move(other.del)),
        valid(other.valid) {
        other.valid = false;
    }

    BoundResource& operator=(BoundResource&& other) {
        if (this == &other)
            return *this;
        close();
        parent = std::move(other.parent);
        value = std::move(other.value);
        data = std::move(other.data);
        eq = std::move(other.eq);
        gen = std::move(other.gen);
        del = std::move(other.del);
        valid = other.valid;
        other.valid = false;
        return *this;
    }

    ~BoundResource() { close(); }

    T& bind(const P& parentResource) {
        if (!valid || !eq(data, parent, parentResource, value)) {
            close();
            parent = parentResource;
            value = gen(data, parent);
            valid = true;
        }
        return get();
    }

    const T& get() const {
        drv::drv_assert(valid, "Resource is not bound");
        return value;
    }

    T& get() {
        drv::drv_assert(valid, "Resource is not bound");
        return value;
    }

 private:
    P parent;
    T value;
    D data;
    Eq eq;
    Gen gen;
    Del del;
    bool valid = false;

    void close() {
        if (valid) {
            del(data, std::move(value));
            valid = false;
        }
    }
};

}  // namespace drv
