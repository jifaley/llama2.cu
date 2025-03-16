// 新增文件 device_code.cu
#include <cuda_runtime.h>
#include "device_code.h"
#include <device_launch_parameters.h>
#include <algorithm>

static float* d_x = nullptr;  // 设备端输入缓存
static float* d_w = nullptr;  // 设备端权重缓存
static float* d_out = nullptr;// 设备端输出缓存

static int current_max_dim = 0;

void cuda_matmul_init(int required_dim) {
	if (required_dim > current_max_dim) { // 仅当需要更大内存时重新分配
		if (d_x) cudaFree(d_x);
		if (d_w) cudaFree(d_w);
		if (d_out) cudaFree(d_out);

		size_t matrix_size = required_dim * required_dim;
		cudaMalloc(&d_x, required_dim * sizeof(float));
		cudaMalloc(&d_w, matrix_size * sizeof(float));
		cudaMalloc(&d_out, required_dim * sizeof(float));

		current_max_dim = required_dim;
	}
}

__global__ void matmul_kernel(float* out, const float* x, const float* w, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d) {
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		out[i] = val;
	}
}

//void cuda_matmul(float* xout, float* x, float* w, int n, int d) {
//	int required_dim = std::max(n, d);
//	if (required_dim > current_max_dim) {
//		cuda_matmul_init(required_dim); // 动态扩展内存
//	}
//
//	// ...原有数据传输和kernel调用代码...
//}

void cuda_matmul(float* xout, float* x, float* w, int n, int d) {

	int required_dim = std::max(n, d);
	if (required_dim > current_max_dim) {
		cuda_matmul_init(required_dim); // 动态扩展内存
	}

	// 拷贝输入数据到设备
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, w, n * d * sizeof(float), cudaMemcpyHostToDevice);

	// 调用kernel
	const int block_size = 256;
	int grid_size = (d + block_size - 1) / block_size;
	matmul_kernel << <grid_size, block_size >> > (d_out, d_x, d_w, n, d);

	// 拷贝结果回主机
	cudaMemcpy(xout, d_out, d * sizeof(float), cudaMemcpyDeviceToHost);
}
