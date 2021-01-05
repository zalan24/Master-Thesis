#include "material.h"

Material::Material(Channel<RGBA>&& aa) : albedo_alpha(channel_to_tex(std::move(aa))) {
}
