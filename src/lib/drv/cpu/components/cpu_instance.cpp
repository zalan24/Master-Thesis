#include "drvcpu.h"

#include "cpu.h"

drv::InstancePtr drv_cpu::create_instance(const drv::InstanceCreateInfo*) {
    return reinterpret_cast<drv::InstancePtr>(new CpuContext);
}

bool drv_cpu::delete_instance(drv::InstancePtr ptr) {
    delete reinterpret_cast<CpuContext*>(ptr);
    return true;
}
