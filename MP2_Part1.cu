//David Krayacich 20381405
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#define TILE_WIDTH 15

float test_res[4];
int rep = 0;

__global__ void matMulKernel(float* M, float* N, float* P, int Width) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	// loop through phases
	for (int ph = 0; ph < Width / TILE_WIDTH; ph++) {
		// each thread loads one element into shared memory
		Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
		Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

		//sync threads here to ensure everything that is needed for this phase is loaded into shared memory
		__syncthreads();

		// perform matrix mult. for that tile
		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}

	P[Row * Width + Col] = Pvalue;
	// written up to here
}

void matMul(float* M, float* N, float* P, int Width) {
	int size = Width * Width * sizeof(float);
	float* d_M;
	float* d_N;
	float* d_P;

	cudaMalloc((void**)&d_M, size);
	cudaMalloc((void**)&d_N, size);
	cudaMalloc((void**)&d_P, size);

	cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

	// im pretty sure in this case I just replace block_width with tile_width and it will work!
	int NumBlocks = Width / TILE_WIDTH;
	if (Width % TILE_WIDTH) NumBlocks++;

	dim3 dimGrid(NumBlocks, NumBlocks);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize();

	cudaEventRecord(start, 0);
	matMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, Width); //for regular matrix mult
	//matMulKernelB << <1, 1 >> > (d_M, d_N, d_P, Width); //for case of 1 block with with one thread
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	test_res[rep] = time;
	rep++;
	printf("Mat Mul kernel Time (ms): %f \n", time);

	cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}

int main(int argc, char* argv[]) {
	int Width = 300;
	int size = Width * Width * sizeof(float);
	float* M1 = (float*)malloc(size * sizeof(float));
	float* N1 = (float*)malloc(size * sizeof(float));
	float* M2 = (float*)malloc(size * sizeof(float));
	float* N2 = (float*)malloc(size * sizeof(float));
	float* M3 = (float*)malloc(size * sizeof(float));
	float* N3 = (float*)malloc(size * sizeof(float));
	float* M4 = (float*)malloc(size * sizeof(float));
	float* N4 = (float*)malloc(size * sizeof(float));
	float* P_g1 = (float*)calloc(size, sizeof(float)); //size * sizeof(float)
	float* P_g2 = (float*)calloc(size, sizeof(float)); //size * sizeof(float)
	float* P_g3 = (float*)calloc(size, sizeof(float)); //size * sizeof(float)
	float* P_g4 = (float*)calloc(size, sizeof(float)); //size * sizeof(float)
	float* P_c1 = (float*)calloc(size, sizeof(float));

	for (int i = 0; i < Width * Width; i++) {
		M1[i] = ((float)rand() / RAND_MAX) * 100; //make random floating point numbers between 0 and 100
		N1[i] = ((float)rand() / RAND_MAX) * 100;
		M2[i] = ((float)rand() / RAND_MAX) * 100; //make random floating point numbers between 0 and 100
		N2[i] = ((float)rand() / RAND_MAX) * 100;
		M3[i] = ((float)rand() / RAND_MAX) * 100; //make random floating point numbers between 0 and 100
		N3[i] = ((float)rand() / RAND_MAX) * 100;
		M4[i] = ((float)rand() / RAND_MAX) * 100; //make random floating point numbers between 0 and 100
		N4[i] = ((float)rand() / RAND_MAX) * 100;
	}

	matMul(M1, N1, P_g1, Width);
	matMul(M2, N2, P_g2, Width);
	matMul(M3, N3, P_g3, Width);
	matMul(M4, N4, P_g4, Width);

	printf("Mat Mul time (ms): %f, %f, %f, %f \n", test_res[0], test_res[1], test_res[2], test_res[3]);

	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			for (int k = 0; k < Width; k++) {
				P_c1[i * Width + j] += M1[i * Width + k] * N1[k * Width + j];
			}
		}
	}

	cudaDeviceSynchronize();
	int failed = 0;
	for (int i = 0; i < Width * Width; i++) {
		//printf("CPU: %f, GPU: %f \n", P_c1[i], P_g1[i]);
		if (abs(P_c1[i] - P_g1[i]) > 0.1) {
			failed = 1;
		}
	}
	if (failed == 0) {
		printf("Test PASSED");
	}
}
