//David Krayacich 20381405
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#define BLOCK_WIDTH 16

__global__ void matMulKernel(float* M, float* N, float* P, int Width) {

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if (Row < Width && Col < Width) {
		float Pvalue = 0;

		for (int k = 0; k < Width; ++k) {
			Pvalue += M[Row * Width + k] * N[k * Width + Col];
		}
		P[Row * Width + Col] = Pvalue;
	}

}


void matMul(float* M, float* N, float* P, int Width) {
	int size = Width * Width * sizeof(float);
	float* d_M;
	float* d_N;
	float* d_P;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize();

	cudaMalloc((void**)&d_M, size);
	cudaMalloc((void**)&d_N, size);
	cudaMalloc((void**)&d_P, size);

	cudaEventRecord(start, 0);
	cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Host to Dev transfer time (ms): %f \n", time);

	int NumBlocks = Width / BLOCK_WIDTH;
	if (Width % BLOCK_WIDTH) NumBlocks++;

	dim3 dimGrid(NumBlocks, NumBlocks);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

	matMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, Width);

	cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}

int main(int argc, char* argv[]) {
	int Width = 1500;
	int size = Width * Width * sizeof(float);
	float* M = (float*)malloc(size * sizeof(float));
	float* N = (float*)malloc(size * sizeof(float));
	float* P_g = (float*)calloc(size, sizeof(float)); //size * sizeof(float)
	float* P_c = (float*)calloc(size, sizeof(float));

	for (int i = 0; i < Width * Width; i++) {
		M[i] = ((float)rand() / RAND_MAX) * 100; //make random floating point numbers between 0 and 100
		N[i] = ((float)rand() / RAND_MAX) * 100;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize();

	matMul(M, N, P_g, Width);

	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			for (int k = 0; k < Width; k++) {
				P_c[i * Width + j] += M[i * Width + k] * N[k * Width + j];
			}
		}
	}

	cudaDeviceSynchronize();
	int failed = 0;
	for (int i = 0; i < Width*Width; i++) {
		//printf("CPU: %f, GPU: %f \n", P_c[i], P_g[i]);
		if (abs(P_c[i] - P_g[i]) > 0.0001) {
			failed = 1;
		}
	}
	if (failed == 0) {
		printf("Test PASSED");
	}
}




