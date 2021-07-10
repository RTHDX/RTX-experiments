cmake_minimum_required (VERSION 3.18)
project(renderer LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} Render.cpp Material.cpp Light.cpp Hit.cpp Ray.cpp
            Scene.cpp Camera.cpp)
target_link_libraries(${PROJECT_NAME} GLAD ${OPENGL_LIBRARIES} glfw)