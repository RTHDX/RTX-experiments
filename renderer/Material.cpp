#include "Material.hpp"

namespace render {

Material::Material(Color color, Albedo albedo, float specular_exponent)
    : color(std::move(color))
    , albedo(std::move(albedo))
    , specular_exponent(specular_exponent)
{}

}