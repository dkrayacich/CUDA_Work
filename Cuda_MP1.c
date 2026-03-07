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
			"Global Mem: %d \n"
			"Constant Mem: %d \n"
			"Shared Mem Per Block: %d \n"
			"Num Regs Available Per Block: %d \n"
			"Max Threads Per Block: %d \n"
			"Block X Dim: %d \n"
			"Block Y DIm: %d \n"
			"Block Z Dim: %d \n"
			"Grid X Dim: %d \n"
			"Grid Y Dim: %d \n"
			"Grid Z Dim: %d \n",
			d, dp.clockRate, dp.multiProcessorCount);
	}
}

