//David Krayacich 20381405
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#define BLOCK_WIDTH 16

int main(int argc, char* argv[]) {
	int dev_count;
	cudaError_t e = cudaGetDeviceCount(&dev_count);
	printf("Device Count: %d \n", dev_count);

	for (int d = 0; d < dev_count; d++) {
		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, d);

		printf("Device %s --> Clock Rate: %d \n"
			"Num SMs: %d \n"
			"Num Cores: 4864 \n" //this number was googled based on the type of GPU as it was not included in the deviceProp Struct
			"Warp Size: %d \n"
			"Global Mem (bytes): %zu \n"
			"Constant Mem (bytes): %zu \n"
			"Shared Mem Per Block (bytes): %zu \n"
			"Num Regs Available Per Block: %d \n"
			"Max Threads Per Block: %d \n"
			"Max Block X Dim: %d \n"
			"Max Block Y DIm: %d \n"
			"Max Block Z Dim: %d \n"
			"Max Grid X Dim: %d \n"
			"Max Grid Y Dim: %d \n"
			"Max Grid Z Dim: %d \n",
			dp.name, dp.clockRate, dp.multiProcessorCount, dp.warpSize, dp.totalGlobalMem,
			dp.totalConstMem, dp.sharedMemPerBlock, dp.regsPerBlock, dp.maxThreadsPerBlock,
			dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2], dp.maxGridSize[0],
			dp.maxGridSize[1], dp.maxGridSize[2]);



	}
}
