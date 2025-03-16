// 新增文件 device_code.h
#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

	void cuda_matmul_init(int max_dim);
	void cuda_matmul(float* xout, float* x, float* w, int n, int d);
	void cuda_memcpy_to_device(const float* host_ptr, size_t size);
	void cuda_memcpy_to_host(float* host_ptr, size_t size);

#ifdef __cplusplus
}
#endif
