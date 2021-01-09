#include "material.h"

Material::Material(DiffuseChannel&& aa) : albedo_alpha(std::move(aa)) {
}

const Material::DiffuseTexRef& Material::getAlbedoAlpha() const {
    if (std::holds_alternative<DiffuseTexRef>(albedo_alpha))
        return std::get<DiffuseTexRef>(albedo_alpha);
    return std::get<DiffuseRes>(albedo_alpha).getRes();
}
