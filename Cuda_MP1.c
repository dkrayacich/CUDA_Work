#include "cuda_runtime.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
	int dev_count;
	cudaError_t e = cudaGetDeviceCount(&dev_count);
	printf("Device Count: %d \n", dev_count);

	for (int d = 0; d < dev_count; d++) {
		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, d);

		printf("Device %d --> Clock Rate: %d \n"
			"Num SMs: %d \n"
			"Num Cores: %d \n"
			"Warp Size: %d \n"
			"Global Mem: %zu \n"
			"Constant Mem: %zu \n"
			"Shared Mem Per Block: %zu \n"
			"Num Regs Available Per Block: %d \n"
			"Max Threads Per Block: %d \n"
			"Block X Dim: %d \n"
			"Block Y DIm: %d \n"
			"Block Z Dim: %d \n"
			"Grid X Dim: %d \n"
			"Grid Y Dim: %d \n"
			"Grid Z Dim: %d \n",
			d, dp.clockRate, dp.multiProcessorCount, d, dp.warpSize, dp.totalGlobalMem,
			dp.totalConstMem, dp.sharedMemPerBlock, dp.regsPerBlock, dp.maxThreadsPerBlock,
			dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2], dp.maxGridSize[0],
			dp.maxGridSize[1], dp.maxGridSize[2]);

	}
}


void matMult()
