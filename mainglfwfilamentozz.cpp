#define _ITERATOR_DEBUG_LEVEL 0

#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <filament/SkinningBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <filament/Renderer.h>

#include <utils/EntityManager.h>

#include <filagui/ImGuiHelper.h>

// ozz-animation headers
#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/base/io/stream.h"
#include "ozz/base/io/archive.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/maths/vec_float.h"
#include "ozz/util/mesh.h"

#include "HandmadeMath.h"

#include <cmath>

/**
 * matc -o bakedColor.filament -a all bakedColor.mat
 * bin2header.exe -o bakedColor.filament.h bakedColor.filament
 */
#include "bakedColor.filament.h"

#define YAP_ENABLE
#define YAP_IMPL
#define YAP_IMGUI
#include "imgui.h"
#include "YAP.h"

#include "imgui_impl_glfw.h"

#define SCREEN_WIDTH  800
#define SCREEN_HEIGHT 600

static std::vector<filament::math::mat4f> convert_ozzMat4_to_filaMat4(const std::vector<ozz::math::Float4x4>& inOzzMat4V)
{
    std::vector<filament::math::mat4f> outFMat4V;
    outFMat4V.resize(inOzzMat4V.size());

    memcpy(outFMat4V.data(), inOzzMat4V.data(), sizeof(float) * 16 * inOzzMat4V.size());

    return outFMat4V;
}

int main()
{
    YAP::Init(2, 4, 2048, 16);// , malloc, free);
    YAP::PushPhase(LoadingPhase);
    YAP::PushSection(Loading);

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "GLFW + Filament + ozz", nullptr, nullptr);

    // ozz init & vb/ib init
#pragma region ozz init
    struct ozz_t
    {
        ozz::animation::Skeleton skeleton;
        ozz::animation::Animation animation;
        ozz::vector<uint16_t> joint_remaps;
        ozz::vector<ozz::math::Float4x4> mesh_inverse_bindposes;
        ozz::vector<ozz::math::SoaTransform> local_matrices;
        ozz::vector<ozz::math::Float4x4> model_matrices;
        ozz::animation::SamplingCache cache;
        std::vector<ozz::math::Float4x4> joint_matrices;

        std::vector<filament::math::mat4f> joint_matrices_fmath;
    };

    struct Vertex
    {
        HMM_Vec3 position;
        HMM_Vec3 normal;
        uint16_t joint_indices[4];
        HMM_Vec4 joint_weights;
    };

    ozz_t ozz{};
    int num_skeleton_joints = -1;
    int num_skin_joints = -1;
    int num_triangle_indices = -1;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    ozz::io::File skel_file("data/ozz_skin_skeleton.ozz", "rb");
    if (!skel_file.opened()) {
        return -1;
    }
    else
    {
        ozz::io::IArchive archive(&skel_file);
        if (archive.TestTag<ozz::animation::Skeleton>()) {
            archive >> ozz.skeleton;

            const int num_soa_joints = ozz.skeleton.num_soa_joints();
            const int num_joints = ozz.skeleton.num_joints();
            ozz.local_matrices.resize(num_soa_joints);
            ozz.model_matrices.resize(num_joints);
            num_skeleton_joints = num_joints;
            ozz.cache.Resize(num_joints);
        }
        else {
            return -1;
        }
    }


    ozz::io::File anim_file("data/ozz_skin_animation.ozz", "rb");
    if (!anim_file.opened()) {
        return -1;
    }
    else
    {
        ozz::io::IArchive archive(&anim_file);
        if (archive.TestTag<ozz::animation::Animation>()) {
            archive >> ozz.animation;
        }
        else {
            return -1;
        }
    }


    ozz::io::File mesh_file("data/ozz_skin_mesh.ozz", "rb");
    if (!mesh_file.opened()) {
        return -1;
    }
    else
    {
        ozz::vector<ozz::sample::Mesh> meshes;
        ozz::io::IArchive archive(&mesh_file);
        while (archive.TestTag<ozz::sample::Mesh>()) {
            meshes.resize(meshes.size() + 1);
            archive >> meshes.back();
        }
        // assume one mesh and one submesh
        assert((meshes.size() == 1) && (meshes[0].parts.size() == 1));
        num_skin_joints = meshes[0].num_joints();
        num_triangle_indices = (int)meshes[0].triangle_index_count();
        ozz.joint_remaps = std::move(meshes[0].joint_remaps);
        ozz.mesh_inverse_bindposes = std::move(meshes[0].inverse_bind_poses);

        // convert mesh data into packed vertices
        size_t num_vertices = (meshes[0].parts[0].positions.size() / 3);
        assert(meshes[0].parts[0].normals.size() == (num_vertices * 3));
        assert(meshes[0].parts[0].joint_indices.size() == (num_vertices * 4));
        assert(meshes[0].parts[0].joint_weights.size() == (num_vertices * 3));
        const float* positions = &meshes[0].parts[0].positions[0];
        const float* normals = &meshes[0].parts[0].normals[0];
        const uint16_t* joint_indices = &meshes[0].parts[0].joint_indices[0];
        const float* joint_weights = &meshes[0].parts[0].joint_weights[0];
        vertices.resize(num_vertices);
        for (int i = 0; i < (int)num_vertices; i++) {
            Vertex* v = &vertices[i];
            v->position[0] = positions[i * 3 + 0];
            v->position[1] = positions[i * 3 + 1];
            v->position[2] = positions[i * 3 + 2];
            v->normal[0] = normals[i * 3 + 0];
            v->normal[1] = normals[i * 3 + 1];
            v->normal[2] = normals[i * 3 + 2];
            v->joint_indices[0] = joint_indices[i * 4 + 0];
            v->joint_indices[1] = joint_indices[i * 4 + 1];
            v->joint_indices[2] = joint_indices[i * 4 + 2];
            v->joint_indices[3] = joint_indices[i * 4 + 3];
            const float jw0 = joint_weights[i * 3 + 0];
            const float jw1 = joint_weights[i * 3 + 1];
            const float jw2 = joint_weights[i * 3 + 2];
            const float jw3 = 1.0f - (jw0 + jw1 + jw2);
            v->joint_weights[0] = std::max(jw0, 0.0f);
            v->joint_weights[1] = std::max(jw1, 0.0f);
            v->joint_weights[2] = std::max(jw2, 0.0f);
            v->joint_weights[3] = std::max(jw3, 0.0f);
        }

        indices.resize(num_triangle_indices);
        for (int idx = 0; idx < num_triangle_indices; ++idx) {
            indices[idx] = meshes[0].triangle_indices[idx];
        }

        ozz.joint_matrices.resize(num_skin_joints);
        ozz.joint_matrices_fmath = convert_ozzMat4_to_filaMat4(ozz.joint_matrices);
    }

#pragma endregion ozz init

    filament::Engine* engine = filament::Engine::create(filament::Engine::Backend::VULKAN);
    filament::SwapChain* swapChain = engine->createSwapChain(glfwGetWin32Window(window));
    filament::Renderer* renderer = engine->createRenderer();

    utils::Entity cameraComponent = utils::EntityManager::get().create();
    filament::Camera* camera = engine->createCamera(cameraComponent);

    filament::View* view = engine->createView();
    view->setName("view0");
    view->setViewport({ 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT });
    view->setPostProcessingEnabled(false);

    filament::Scene* scene = engine->createScene();

    filament::Skybox* skybox = filament::Skybox::Builder().color({0.1, 0.125, 0.25, 1.0}).build(*engine);
    scene->setSkybox(skybox);

    filament::Material* material = filament::Material::Builder()
            .package((void*) bakedColor_filament, sizeof(bakedColor_filament))
            .build(*engine);
    filament::MaterialInstance* materialInstance = material->createInstance();

    auto l = vertices[vertices.size() - 1];

    filament::VertexBuffer* vertexBuffer = filament::VertexBuffer::Builder()
            .vertexCount(vertices.size())
            .bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION,     0, filament::VertexBuffer::AttributeType::FLOAT3, 0,  48)
            .attribute(filament::VertexAttribute::COLOR,        0, filament::VertexBuffer::AttributeType::FLOAT3, 12, 48)
//            .normalized(filament::VertexAttribute::COLOR)
            .attribute(filament::VertexAttribute::BONE_INDICES, 0, filament::VertexBuffer::AttributeType::USHORT4, 24, 48)
            .attribute(filament::VertexAttribute::BONE_WEIGHTS, 0, filament::VertexBuffer::AttributeType::FLOAT4,  32, 48)
            .build(*engine);
    vertexBuffer->setBufferAt(*engine, 0, filament::VertexBuffer::BufferDescriptor(vertices.data(), sizeof(Vertex)*vertices.size(), nullptr));

    filament::IndexBuffer* indexBuffer = filament::IndexBuffer::Builder()
            .indexCount(indices.size())
            .bufferType(filament::IndexBuffer::IndexType::UINT)
            .build(*engine);
    indexBuffer->setBuffer(*engine, filament::IndexBuffer::BufferDescriptor(indices.data(), sizeof(uint32_t)*indices.size(), nullptr));

    utils::Entity renderable = utils::EntityManager::get().create();
    // build a triangle
    filament::RenderableManager::Builder(1)
            .boundingBox({{ -1, -1, -1 }, { 1, 1, 1 }})
            .material(0, materialInstance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES, vertexBuffer, indexBuffer, 0, indices.size())
            .culling(false)
            .receiveShadows(false)
            .castShadows(false)
//            .enableSkinningBuffers(false)
            .skinning(ozz.joint_matrices_fmath.size())
            .build(*engine, renderable);
    scene->addEntity(renderable);
    auto& tcm = engine->getTransformManager();
//    tcm.setTransform(tcm.getInstance(renderable), filament::math::mat4f::rotation(180.0f * HMM_DegToRad, filament::math::float3{ 0, 1, 0 }));
    tcm.setTransform(tcm.getInstance(renderable), filament::math::mat4f::translation(filament::math::float3{ 0, -1, -1 }));

    view->setCamera(camera);
    view->setScene(scene);

    // imgui
    filament::View* ui_view = engine->createView();
    ui_view->setName("ui_view0");
    ui_view->setViewport({ 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT });
    ui_view->setPostProcessingEnabled(false);
    filagui::ImGuiHelper* im_gui_helper = new filagui::ImGuiHelper{engine, ui_view, ""};
    const auto size = ui_view->getViewport();
    im_gui_helper->setDisplaySize(size.width, size.height, 1, 1);
    ImGui_ImplGlfw_InitForOther(window, true);


    YAP::PopSection();
    YAP::PopPhase(); // LoadingPhase

    YAP::PushPhase(GamePhase);

    static double startTime = glfwGetTime();
    bool enableYAP = true;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, true);
        }
        float now = glfwGetTime() - startTime;
        ImGui_ImplGlfw_NewFrame();

        // beginFrame() returns false if we need to skip a frame
        if (renderer->beginFrame(swapChain)) {
            YAP::NewFrame();
            YAP::PushSection(Frame);

            // camera
            {
                YAP::PushSection(CameraUpdate, 0XFF000099);
                constexpr float ZOOM = 1.5f;
                const uint32_t w = view->getViewport().width;
                const uint32_t h = view->getViewport().height;
                const float aspect = (float) w / h;
                camera->setProjection(filament::Camera::Projection::ORTHO,
                                       -aspect * ZOOM, aspect * ZOOM,
                                       -ZOOM, ZOOM, 0, 1);
                YAP::PopSection();
            }

            // ozz update
            {
                YAP::PushSection(OzzUpdate, 0XFF009900);
                // convert current time to animation ration (0.0 .. 1.0)
                const float anim_duration = ozz.animation.duration();
                float anim_ratio = fmodf((float)now / anim_duration, 1.0f);

                // sample animation
                YAP::PushSection(SamplingJob, 0XFF999900);
                ozz::animation::SamplingJob sampling_job;
                sampling_job.animation = &ozz.animation;
                sampling_job.cache = &ozz.cache;
                sampling_job.ratio = anim_ratio;
                sampling_job.output = make_span(ozz.local_matrices);
                sampling_job.Run();
                YAP::PopSection(); // SamplingJob

                // convert joint matrices from local to model space
                YAP::PushSection(LocalToModelJob, 0XFF009999);
                ozz::animation::LocalToModelJob ltm_job;
                ltm_job.skeleton = &ozz.skeleton;
                ltm_job.input = make_span(ozz.local_matrices);
                ltm_job.output = make_span(ozz.model_matrices);
                ltm_job.Run();
                YAP::PopSection(); // LocalToModelJob

                // compute skinning matrices and write to joint texture upload buffer
                YAP::PushSection(ComputeSkinningMat, 0XFF990099);
                for (int i = 0; i < num_skin_joints; i++) {
                    ozz.joint_matrices[i] = ozz.model_matrices[ozz.joint_remaps[i]] * ozz.mesh_inverse_bindposes[i];
                }

                ozz.joint_matrices_fmath = convert_ozzMat4_to_filaMat4(ozz.joint_matrices);

                auto& rm = engine->getRenderableManager();
                rm.setBones(rm.getInstance(renderable), ozz.joint_matrices_fmath.data(), ozz.joint_matrices_fmath.size(), 0);
                YAP::PopSection(); // ComputeSkinningMat

                YAP::PopSection(); // OzzUpdate
            }

            // for each View
            YAP::PushSection(ImguiUpdate, 0XFF567800);
            im_gui_helper->render(now, [&enableYAP](filament::Engine* engine, filament::View* view) {
                // ImGui::ShowDemoWindow();
                YAP::PushSection(YapUpdate, 0XFF654321);
                YAP::ImGuiLogger(&enableYAP);
                YAP::PopSection(); // YapUpdate
            });
            YAP::PopSection(); // ImguiUpdate

            YAP::PopSection(); // Frame

            // YAP::PushSection(CmdbufferUpdate, 0XFF009876);
            renderer->render(view);
            renderer->render(ui_view);
            // YAP::PopSection(); // CmdbufferUpdate, 0XFF567800);

            // YAP::PushSection(Present, 0XFF123456);
            renderer->endFrame();
            // YAP::PopSection(); // Present, 0XFF567800);

        }
    }

    YAP::PopPhase(); // GamePhase
    YAP::LogDump(printf);
    YAP::Finish();

    ImGui_ImplGlfw_Shutdown();
    delete im_gui_helper;
    engine->destroy(scene);
    engine->destroy(skybox);
    engine->destroy(renderable);
    engine->destroy(materialInstance);
    engine->destroy(material);
//    engine->destroy(skinningBuffer);
    engine->destroy(vertexBuffer);
    engine->destroy(indexBuffer);
    engine->destroy(swapChain);
    engine->destroy(renderer);
    engine->destroyCameraComponent(cameraComponent);
    utils::EntityManager::get().destroy(cameraComponent);
    engine->destroy(view);
    engine->destroy(ui_view);
    filament::Engine::destroy(engine);

    glfwTerminate();
    return EXIT_SUCCESS;
}
