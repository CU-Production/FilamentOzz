#define _ITERATOR_DEBUG_LEVEL 0
#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/SkinningBuffer.h>
#include <filament/View.h>

#include <utils/EntityManager.h>
#include <utils/Path.h>

#include <filamentapp/Config.h>
#include <filamentapp/FilamentApp.h>

#include <cmath>
#include <iostream>

#include <getopt/getopt.h>

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

#include "bakedColor.filament.h"

using namespace filament;
using utils::Entity;
using utils::EntityManager;
using utils::Path;
using namespace filament::math;

struct App {
    VertexBuffer* vb;
    IndexBuffer* ib;
    Material* mat;
    Camera* cam;
    Entity camera;
    Skybox* skybox;
    Entity renderable;
};

struct VertexWithBones {
    float3 position;
    float3 normal;
    filament::math::ushort4 joint_indices;
    float4 joint_weights;
};

static std::vector<filament::math::mat4f> convert_ozzMat4_to_filaMat4(const std::vector<ozz::math::Float4x4>& inOzzMat4V)
{
    std::vector<filament::math::mat4f> outFMat4V;
    outFMat4V.resize(inOzzMat4V.size());

    memcpy(outFMat4V.data(), inOzzMat4V.data(), sizeof(float) * 16 * inOzzMat4V.size());

//    for (int i = 0; i < outFMat4V.size(); ++i)
//    {
//        outFMat4V[i] = filament::math::mat4f(1.0f);
//    }

    return outFMat4V;
}

static void printUsage(char* name) {
    std::string exec_name(Path(name).getName());
    std::string usage(
            "SAMPLE is a command-line tool for testing Filament skinning.\n"
            "Usage:\n"
            "    SAMPLE [options]\n"
            "Options:\n"
            "   --help, -h\n"
            "       Prints this message\n\n"
            "   --api, -a\n"
            "       Specify the backend API: opengl (default), vulkan, or metal\n\n"
    );
    const std::string from("SAMPLE");
    for (size_t pos = usage.find(from); pos != std::string::npos; pos = usage.find(from, pos)) {
        usage.replace(pos, from.length(), exec_name);
    }
    std::cout << usage;
}

static int handleCommandLineArgments(int argc, char* argv[], Config* config) {
    static constexpr const char* OPTSTR = "ha:";
    static const struct option OPTIONS[] = {
            { "help",         no_argument,       nullptr, 'h' },
            { "api",          required_argument, nullptr, 'a' },
            { nullptr, 0, nullptr, 0 }  // termination of the option list
    };
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, OPTSTR, OPTIONS, &option_index)) >= 0) {
        std::string arg(optarg != nullptr ? optarg : "");
        switch (opt) {
            default:
            case 'h':
                printUsage(argv[0]);
                exit(0);
            case 'a':
                if (arg == "opengl") {
                    config->backend = Engine::Backend::OPENGL;
                } else if (arg == "vulkan") {
                    config->backend = Engine::Backend::VULKAN;
                } else if (arg == "metal") {
                    config->backend = Engine::Backend::METAL;
                } else {
                    std::cerr << "Unrecognized backend. Must be 'opengl'|'vulkan'|'metal'."
                              << std::endl;
                }
                break;
        }
    }

    return optind;
}

int main(int argc, char** argv) {
    Config config;
    config.title = "hello skinning";

    handleCommandLineArgments(argc, argv, &config);

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

//    struct Vertex
//    {
//        HMM_Vec3 position;
//        HMM_Vec3 normal;
//        uint16_t joint_indices[4];
//        HMM_Vec4 joint_weights;
//    };

    ozz_t ozz{};
    int num_skeleton_joints = -1;
    int num_skin_joints = -1;
    int num_triangle_indices = -1;
    std::vector<VertexWithBones> vertices;
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
            VertexWithBones* v = &vertices[i];
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
            v->joint_weights[0] = jw0;
            v->joint_weights[1] = jw1;
            v->joint_weights[2] = jw2;
            v->joint_weights[3] = jw3;
        }

        indices.resize(num_triangle_indices);
        for (int idx = 0; idx < num_triangle_indices; ++idx) {
            indices[idx] = meshes[0].triangle_indices[idx];
        }

        ozz.joint_matrices.resize(num_skin_joints);
        ozz.joint_matrices_fmath = convert_ozzMat4_to_filaMat4(ozz.joint_matrices);
    }

#pragma endregion ozz init

    App app;
    auto setup = [&app, &vertices, &indices, &ozz](Engine* engine, View* view, Scene* scene) {
        app.skybox = Skybox::Builder().color({0.1, 0.125, 0.25, 1.0}).build(*engine);

        scene->setSkybox(app.skybox);
        view->setPostProcessingEnabled(false);
        static_assert(sizeof(VertexWithBones) == 48, "Strange vertex size.");
        app.vb = VertexBuffer::Builder()
                .vertexCount(vertices.size())
                .bufferCount(1)
                .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT3, 0, 48)
                .attribute(VertexAttribute::COLOR, 0, VertexBuffer::AttributeType::FLOAT3, 12, 48)
//                .normalized(VertexAttribute::COLOR)
                .attribute(VertexAttribute::BONE_INDICES, 0, VertexBuffer::AttributeType::USHORT4, 24, 48)
                .attribute(VertexAttribute::BONE_WEIGHTS, 0, VertexBuffer::AttributeType::FLOAT4, 32, 48)
                .build(*engine);
        app.vb->setBufferAt(*engine, 0,
                            VertexBuffer::BufferDescriptor(vertices.data(), sizeof(VertexWithBones)*vertices.size(), nullptr));
        app.ib = IndexBuffer::Builder()
                .indexCount(indices.size())
                .bufferType(IndexBuffer::IndexType::UINT)
                .build(*engine);
        app.ib->setBuffer(*engine,
                          IndexBuffer::BufferDescriptor(indices.data(), sizeof(uint32_t)*indices.size(), nullptr));
        app.mat = Material::Builder()
                .package(bakedColor_filament, sizeof(bakedColor_filament))
                .build(*engine);

        app.renderable = EntityManager::get().create();

        RenderableManager::Builder(1)
                .boundingBox({{ -1, -1, -1 }, { 1, 1, 1 }})
                .material(0, app.mat->getDefaultInstance())
                .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, app.vb, app.ib, 0, 1500)
                .culling(false)
                .receiveShadows(false)
                .castShadows(false)
                .skinning(ozz.joint_matrices_fmath.size())
                .enableSkinningBuffers(false)
                .build(*engine, app.renderable);

        scene->addEntity(app.renderable);
        auto& tcm = engine->getTransformManager();
        tcm.setTransform(tcm.getInstance(app.renderable), filament::math::mat4f::translation(filament::math::float3{ 0, -1, 0 }));

        app.camera = utils::EntityManager::get().create();
        app.cam = engine->createCamera(app.camera);
        view->setCamera(app.cam);
    };

    auto cleanup = [&app](Engine* engine, View*, Scene*) {
        engine->destroy(app.skybox);
        engine->destroy(app.renderable);
        engine->destroy(app.mat);
        engine->destroy(app.vb);
        engine->destroy(app.ib);
        engine->destroyCameraComponent(app.camera);
        utils::EntityManager::get().destroy(app.camera);
    };

    FilamentApp::get().animate([&app, &ozz](Engine* engine, View* view, double now) {
        constexpr float ZOOM = 1.5f;
        const uint32_t w = view->getViewport().width;
        const uint32_t h = view->getViewport().height;
        const float aspect = (float) w / h;
        app.cam->setProjection(Camera::Projection::ORTHO,
                               -aspect * ZOOM, aspect * ZOOM,
                               -ZOOM, ZOOM, 0, 1);

        // ozz update
        {
            // convert current time to animation ration (0.0 .. 1.0)
            const float anim_duration = ozz.animation.duration();
            float anim_ratio = fmodf((float)now / anim_duration, 1.0f);

            // sample animation
            ozz::animation::SamplingJob sampling_job;
            sampling_job.animation = &ozz.animation;
            sampling_job.cache = &ozz.cache;
            sampling_job.ratio = anim_ratio;
            sampling_job.output = make_span(ozz.local_matrices);
            sampling_job.Run();

            // convert joint matrices from local to model space
            ozz::animation::LocalToModelJob ltm_job;
            ltm_job.skeleton = &ozz.skeleton;
            ltm_job.input = make_span(ozz.local_matrices);
            ltm_job.output = make_span(ozz.model_matrices);
            ltm_job.Run();

            // compute skinning matrices and write to joint texture upload buffer
            for (int i = 0; i < ozz.joint_matrices.size(); i++) {
                ozz.joint_matrices[i] = ozz.model_matrices[ozz.joint_remaps[i]] * ozz.mesh_inverse_bindposes[i];
            }

            ozz.joint_matrices_fmath = convert_ozzMat4_to_filaMat4(ozz.joint_matrices);
        }

        auto& rm = engine->getRenderableManager();

        // Bone skinning animation
        rm.setBones(rm.getInstance(app.renderable), ozz.joint_matrices_fmath.data(), ozz.joint_matrices_fmath.size(), 0);

    });

    FilamentApp::get().run(config, setup, cleanup);

    return 0;
}
