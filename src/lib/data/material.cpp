#include "material.h"

Material::Material() : albedo_alpha(GenericResourcePool::ResourceRef()) {
}

Material::Material(DiffuseChannel&& aa) : albedo_alpha(std::move(aa)) {
}

const Material::DiffuseTexRef& Material::getAlbedoAlpha() const {
    if (std::holds_alternative<DiffuseTexRef>(albedo_alpha))
        return std::get<DiffuseTexRef>(albedo_alpha);
    return std::get<DiffuseRes>(albedo_alpha).getRes();
}

void Material::writeJson(json& out) const {
    if (!std::holds_alternative<DiffuseRes>(albedo_alpha))
        throw std::runtime_error("This material holds dynamic references. It cannot be exported.");
    const DiffuseRes& albedo = std::get<DiffuseRes>(albedo_alpha);
    WRITE_OBJECT(albedo, out);
}

void Material::readJson(const json& in) {
    DiffuseRes albedo;
    READ_OBJECT(albedo, in);
    albedo_alpha = std::move(albedo);
}
