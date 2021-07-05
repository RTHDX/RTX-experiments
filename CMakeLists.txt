project(OPENGL_CUDA LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glad/include")
add_subdirectory(glad)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glfw/include")
add_subdirectory(glfw)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glm/glm")
add_subdirectory(glm)

add_executable(${PROJECT_NAME} main.cpp render.hpp render.cpp utils.hpp)
target_link_libraries(${PROJECT_NAME} GLAD ${OPENGL_LIBRARIES} glfw)
