#pragma once

#include "Utils.hpp"
#include "renderer/CudaRender.cuh"

using namespace utils;
using namespace render;
using namespace render::cuda;

constexpr int DEFAULT_DIM = 20;

static inline std::vector<Sphere> make_spheres() {
    Material material_1(Color(1.0, 0.1, 0.1), Albedo(0.7, 0.09, 0.1), 40.0);
    Material material_2(Color(0.9, 0.9, 0.9), Albedo(0.5, 0.1, 0.1), 30.0);

    return {
        Sphere(Point(0.0, 0.0, 0.0), 2.0f, material_1),
        Sphere(Point(0.0, -100005.0, 0.0), 100000.0, material_2)
    };
}

static inline std::vector<Light> make_lights() {
    return {
        Light(Point(40.0f, 40.0f, 30.0f), 3.0f),
        Light(Point(-40.0f, 40.0f, 30.0f), 3.0f),
    };
}


static inline Scene make_scene(int width = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return Scene(
        make_spheres(), Color(0.1, 0.01, 0.01), make_lights(),
        width, height
    );
}

inline Camera make_camera(int widht = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return Camera(Point(0.0, 0.0, 50.0), to_radian(45),
                  widht, height);
}

inline CudaRender make_render(int width = DEFAULT_DIM, int height = DEFAULT_DIM) {
    return CudaRender(
        make_scene(width, height),
        make_camera(width, height),
        width, height
    );
}


__global__ void spheres_status(Sphere* list, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        const Point& center = list[i].center();
        printf("<Sphere: center - (%f;%f;%f), radius - %f>\n",
            center.x, center.y, center.z, list[i].radius());
    }
}

__global__ void lights_status(Light* list, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        const Point& pos = list[i].position;
        printf("<Light: position - (%f;%f;%f), intensity - %f>\n",
               pos.x, pos.y, pos.z, list[i].intensity);
    }
}


__global__ void camera_status(Camera* camera) {
    assert(camera);
    const Point& pos = camera->position();
    printf("<Camera: position - (%f;%f;%f), WxH(%d x %d)>\n",
           pos.x, pos.y, pos.z, camera->width(), camera->height());
}


__global__ void scene_status(Scene* scene) {
    assert(scene);
    const auto& objects = scene->objects();
    const auto& back = scene->background();
    const auto& lights = scene->lights();
    size_t width = scene->width();
    size_t heights = scene->height();

    printf("<Scene: objects - (\n");
    for (size_t i = 0; i < objects.len; ++i) {
        const auto& center = objects.list[i].center();
        printf("\t- <Sphere: center - (%f;%f;%f), radius - %f>\n",
               center.x, center.y, center.z, objects.list[i].radius());
    }
    printf("), lights - (\n");
    for (size_t i = 0; i < lights.len; ++i) {
        const auto& pos = lights.list[i].position;
        printf("\t- <Light: position - (%f;%f;%f), intensity - %f>\n",
               pos.x, pos.y, pos.z, lights.list[i].intensity);
    }
    printf(") background - (%f;%f;%f)>\n", back.r, back.g, back.b);
}
