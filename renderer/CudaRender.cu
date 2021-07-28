#include <stdio.h>

#include <glad/glad.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gtx/string_cast.hpp>
#include <gtc/type_ptr.hpp>

#include "CudaRender.cuh"


namespace render { namespace cuda {


Scene::Scene(const std::vector<Sphere>& objects, Color background,
             const std::vector<Light>& lights, size_t w, size_t h)
    : _objects(utils::cuda::convert_to_cuda_managed(objects))
    , _background(std::move(background))
    , _lights(utils::cuda::convert_to_cuda_managed(lights))
    , _width(w)
    , _height(h)
{}


ATTRIBS Hit intersects(const Context* ctx, const Ray& ray) {
    Hit hit(false);
    float distance = utils::cuda::positive_infinite();
    const auto& objects = ctx->scene->objects();
    for (size_t index = 0; index < objects.len; ++index) {
        Hit temp = objects.list[index].hit(ray);
        if (temp.is_hitted() && distance > temp.t_near) {
            hit = temp;
            distance = temp.t_near;
        }
    }
    return hit;
}


ATTRIBS Color trace(const Context* ctx, const Ray& ray) {
    Hit hit = intersects(ctx, ray);
    if (!hit.is_hitted()) { return ctx->scene->background(); }

    float diffuse_light_intensity = 0.0f,
          specular_light_intensity = 0.0f;
    const auto& lights = ctx->scene->lights();
    for (size_t i = 0; i < lights.len; ++i) {
        const auto& point = hit.point;
        const auto& normal = hit.normal;
        float exponent = hit.object->material().specular_exponent;
        diffuse_light_intensity += lights.list[i].diffuce_factor(point, normal);
        specular_light_intensity += lights.list[i]
            .specular_factor(point, normal, exponent);
    }

    const Albedo& albedo = hit.object->material().albedo;
    Color diffuse = hit.object->material().color * diffuse_light_intensity *
                    albedo.x;
    Color specular = Color(1.0, 1.0, 1.0) * specular_light_intensity *
                     albedo.y;
    Color total = diffuse + specular;
    return total;
}

__global__ void kernel_render(const Context* ctx, size_t len, Color* frame) {
    const int w_pos = blockIdx.x;
    const int h_pos = threadIdx.x;
    const int width = gridDim.x;
    const int index = w_pos + (width * h_pos);

    if (index >= len) return;
    if (ctx == nullptr) return;
    if (ctx->camera == nullptr || ctx->scene == nullptr) return;

    Color pixel_color = trace(ctx, ctx->camera->emit_ray(h_pos, w_pos));

    frame[index].r = pixel_color.r;
    frame[index].g = pixel_color.g;
    frame[index].b = pixel_color.b;
}


CudaRender::CudaRender(const Scene& scene, const Camera& camera, int width, int height)
    : BaseRender()
    , _scene(utils::cuda::cuda_copy(_scene, scene))
    , _camera(utils::cuda::cuda_copy(_camera, camera))
    , _width(width)
    , _height(height)
    , _len(width * height)
    , _frame(new Color[_len])
    , _cuda_frame_ptr(utils::cuda::cuda_allocate_buffer(_cuda_frame_ptr, _len))
{}

CudaRender::~CudaRender() {
    cudaFree(_scene->objects().list);
    cudaFree(_scene->lights().list);
    cudaFree(_scene);
    cudaFree(_camera);
    cudaFree(_cuda_frame_ptr);
    delete [] _frame;
}

void CudaRender::render() {
    dim3 block(_width);
    dim3 thread(_height);
    Context* dev_ctx;
    HANDLE_ERROR(cudaMallocManaged(&dev_ctx, sizeof (Context)));
    dev_ctx->camera = _camera;
    dev_ctx->scene = _scene;
    kernel_render<<<block, thread>>>(dev_ctx, _len, _cuda_frame_ptr);
    cudaFree(dev_ctx);
}


float* CudaRender::frame() {
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(_frame, _cuda_frame_ptr, _len * sizeof (Color),
                            cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    return glm::value_ptr(*_frame);
}

void CudaRender::draw() {
    BaseRender::draw();
    glDrawPixels(_width, _height, GL_RGB, GL_FLOAT,
                 frame());
}

}}
