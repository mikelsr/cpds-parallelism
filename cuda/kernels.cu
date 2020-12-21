#include <math.h>
#include <float.h>
#include <cuda.h>

#include "kernels.cuh"

__global__ void gpu_Heat (float *h, float *g, int N) {

	// TODO: kernel computation
	//...
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i < N - 1 && j > 0 && j < N - 1) {
		g[i * N + j] =	0.25 * (h[i * N + (j - 1)]	// left
						+ h[i * N + (j + 1)]		// right
						+ h[(i - 1) * N + j]		// top
						+ h[(i + 1) * N + j]);		// bottom
	}
}

__global__ void gpu_ResidualMatrix (float *h, float *g, float *diff_matrix, int N) {
	// This function can be completely assimilated by gpu_Heat
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i < N - 1 && j > 0 && j < N - 1) {
		// just add the dev_diff_matrix parameter to gpu_Heat
		diff_matrix[i * N + j] =	(g[i * N + j] - h[i * N + j]) * 
									(g[i * N + j] - h[i * N + j]);
	}
}

/*
__global__ void gpu_Residual (float *diff_matrix, float *residual) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	sdata[tid] = diff_matrix[i] + diff_matrix[i + blockDim.x];
	__syncthreads();
	for(unsigned int s = blockDim.x / 4; s > WARP_SIZE; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < WARP_SIZE) warpReduce(sdata, tid);
	if (tid == 0) residual[blockIdx.x] = sdata[tid];
}
*/
