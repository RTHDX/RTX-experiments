#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gtx/string_cast.hpp>

#include "Utils.hpp"
#include "CudaRender.cuh"


namespace render { namespace cuda {

Hit::Hit(IObject* object) : object(object) {}


Scene::Scene(Objects object, Color background, Lights lights, size_t w, size_t h)
    : _objects(std::move(object))
    , _background(std::move(background))
    , _lights(std::move(lights))
    , _width(w)
    , _height(h)
{
    printf("Objects.size: %zd, Background: %s, Lights.size: %zd, %zdx%zd\n",
           _objects.size(), glm::to_string(_background).c_str(),
           _lights.size(), _width, _height);
}

__global__ void kernel_render(CudaRender* render) {
    assert(render);

    printf("*******************DEVICE*******************************\n");
    printf("Camera.width: %d, Camera.height: %d\n",
           render->camera().width(),
           render->camera().height());
    printf("Scene.objects.size: %d", render->scene().background().x);
}

CudaRender::CudaRender(Scene scene, Camera camera, int width, int height)
    : BaseRender()
    , _scene(std::move(scene))
    , _camera(std::move(camera))
    , _len(width * height)
    , _frame(new Color[_len])
    , _cuda_frame_ptr(utils::cuda::cuda_allocate(_cuda_frame_ptr, _len))
{}

void CudaRender::render() {
    _dev_ptr = utils::cuda::cuda_copy(_dev_ptr, *this);
    dim3 block(1);
    dim3 thread(1);
    kernel_render<<<block, thread>>>(_dev_ptr);
}

}}
