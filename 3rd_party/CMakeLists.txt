# GLFW
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "shared")
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "glfw examples")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "glfw tests")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "glfw docs")
set(GLFW_INSTALL OFF CACHE INTERNAL "glfw install")
set(GLFW_VULKAN_STATIC OFF CACHE INTERNAL "glfw vulkan")
add_subdirectory(glfw)
set_property(TARGET glfw PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Filament
set(FILAMENT_SUPPORTS_VULKAN ON)
add_subdirectory(filament)

# Ozz animation
add_library(OzzAnim ozz/src/mesh.cc ozz/src/ozz_animation.cc ozz/src/ozz_base.cc)
target_include_directories(OzzAnim PUBLIC ozz/include)
set_property(TARGET OzzAnim PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_compile_definitions(OzzAnim PRIVATE -D_ITERATOR_DEBUG_LEVEL=0)
