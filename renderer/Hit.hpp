#pragma once

#include <ostream>

#include "Aliases.hpp"


namespace render {

class IObject;
struct Hit {
    float t_near    {std::numeric_limits<float>::max()};
    float t_far     {std::numeric_limits<float>::max()};
    Point point     {Point(1.0f, 1.0f, 1.0f)};
    Vector normal   {Point(1.0f, 1.0f, 1.0f)};
    IObject* object {nullptr};

public:
    Hit() = default;
    Hit(IObject* object);

    bool is_hitted() const;
};
std::ostream& operator << (std::ostream& os, const Hit&);

}