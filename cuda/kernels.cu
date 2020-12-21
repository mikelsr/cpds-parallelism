#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {

	// TODO: kernel computation
	//...
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i > 0 && i < (N-1) && j > 0 && j < (N-1)) {
		g[i * N + j] =	0.25 * (h[i * N + (j - 1)]	// left
						+ h[i * N + (j + 1)]		// right
						+ h[(i - 1) * N + j]		// top
						+ h[(i + 1) * N + j]);		// bottom
	}
}
