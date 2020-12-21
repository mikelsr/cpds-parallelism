#define WARP_SIZE 32


__global__ void gpu_Heat (float *h, float *g, int N);
__global__ void gpu_ResidualMatrix (float *h, float *g, float *diff_matrix, int N);
template<unsigned int blockSize> __device__ void warpReduce(volatile int* sdata, int tid);
/*
template __device__ void warpReduce<512>(volatile int* sdata, int tid);
template __device__ void warpReduce<256>(volatile int* sdata, int tid);
template __device__ void warpReduce<128>(volatile int* sdata, int tid);
template __device__ void warpReduce<64>(volatile int* sdata, int tid);
template __device__ void warpReduce<32>(volatile int* sdata, int tid);
template __device__ void warpReduce<16>(volatile int* sdata, int tid);
template __device__ void warpReduce<8>(volatile int* sdata, int tid);
template __device__ void warpReduce<4>(volatile int* sdata, int tid);
template __device__ void warpReduce<2>(volatile int* sdata, int tid);
template __device__ void warpReduce<1>(volatile int* sdata, int tid);
*/

template<unsigned int blockSize> __global__ void gpu_Residual(float *diff_matrix, float *residual);

/*
template __global__ void gpu_Residual<512>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<256>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<128>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<64>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<32>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<16>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<8>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<4>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<2>(float *diff_matrix, float *residual);
template __global__ void gpu_Residual<1>(float *diff_matrix, float *residual);
*/
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tid) {
	sdata[tid] += sdata[tid + 64];
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid +  8];
	sdata[tid] += sdata[tid +  4];
	sdata[tid] += sdata[tid +  2];
	sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void gpu_Residual (float *diff_matrix, float *residual) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	if (blockSize >= 512) {
		if (tid <= 256) sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid <= 128) sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid <= 64) sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	sdata[tid] = diff_matrix[i] + diff_matrix[i + blockDim.x];

	if (tid < WARP_SIZE) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) residual[blockIdx.x] = sdata[tid];
}
