#include "drvcuda.h"

#include "cudactx.h"

drv::InstancePtr drv_cuda::create_instance(const drv::InstanceCreateInfo*) {
    return reinterpret_cast<drv::InstancePtr>(new CudaContext);
}

bool drv_cuda::delete_instance(drv::InstancePtr ptr) {
    delete reinterpret_cast<CudaContext*>(ptr);
    return true;
}
