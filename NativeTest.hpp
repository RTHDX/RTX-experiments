#include <gtest/gtest.h>

#include "NativeRender.hpp"


class ShadowTest : public testing::Test {
public:
    ShadowTest();

public:
    render::Scene scene;
    render::Camera camera;
    render::NativeRender render;
};