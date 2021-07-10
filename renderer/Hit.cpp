#include <gtx/string_cast.hpp>

#include "Hit.hpp"


namespace render {

Hit::Hit(IObject* object) : object(object) {}

bool Hit::is_hitted() const {
    return object != nullptr;
}

std::ostream& operator << (std::ostream& os, const Hit& hit) {
    return os << "<Hit.\n t_near: " << hit.t_near
              << "\n t_far: " << hit.t_far
              << "\n is hitted: " << hit.is_hitted()
              << "\n point: " << glm::to_string(hit.point)
              << "\n normal: " << glm::to_string(hit.normal)
              << "\n object: " << hit.object << ">";
}

}