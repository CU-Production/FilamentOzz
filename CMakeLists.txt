cmake_minimum_required(VERSION 3.28)
project(DXtest)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(3rd_party)

add_executable(FilamentGlfw mainglfwfilament.cpp)
target_link_libraries(FilamentGlfw PRIVATE glfw backend bluegl opengl32 bluevk filabridge filaflat filamat filament backend ibl utils geometry smol-v)

add_executable(FilamentOzz mainglfwfilamentozz.cpp)
target_link_libraries(FilamentOzz PRIVATE glfw backend bluegl opengl32 bluevk filabridge filaflat filamat filament backend ibl utils geometry smol-v OzzAnim)
set_property(TARGET FilamentOzz PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
add_custom_command(
        TARGET FilamentOzz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/data" "${CMAKE_CURRENT_BINARY_DIR}/data/"
        VERBATIM
)

add_executable(FilamentAppOzz mainfilamentozz.cpp)
target_link_libraries(FilamentAppOzz PRIVATE glfw backend bluegl opengl32 bluevk filabridge filaflat filamat filament backend ibl utils geometry smol-v filamentapp OzzAnim)
set_property(TARGET FilamentAppOzz PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
add_custom_command(
        TARGET FilamentAppOzz POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/data" "${CMAKE_CURRENT_BINARY_DIR}/data/"
        VERBATIM
)
