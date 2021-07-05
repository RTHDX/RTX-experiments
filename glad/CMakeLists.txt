project(GLAD)

add_library(${PROJECT_NAME} STATIC
    include/KHR/khrplatform.h
    include/glad/glad.h
    src/glad.c
)