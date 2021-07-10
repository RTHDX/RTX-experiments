cmake_minimum_required (VERSION 3.18)
project(OPENGL_CUDA LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glad/include")
add_subdirectory(glad)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glfw/include")
add_subdirectory(glfw)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glm/glm")
add_subdirectory(glm)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include")
add_subdirectory(googletest)
enable_testing()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/renderer")
add_subdirectory(renderer)


add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} GLAD ${OPENGL_LIBRARIES} glfw renderer)

add_executable("${PROJECT_NAME}-Test" test.hpp test.cpp)
target_link_libraries("${PROJECT_NAME}-Test"
	GLAD ${OPENGL_LIBRARIES} glfw GTest::gtest GTest::gtest_main GTest::gmock
	GTest::gmock_main renderer
)


