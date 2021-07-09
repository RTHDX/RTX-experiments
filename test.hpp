#include <gtest/gtest.h>

#include "render.hpp"


class ShadowTest : public testing::Test {
public:
    ShadowTest();

public:
    render::Scene scene;
    render::Camera camera;
    render::Render render;
};