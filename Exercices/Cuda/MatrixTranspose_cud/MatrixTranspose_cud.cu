#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <assert.h>

#define N 8192
#define BLOCK_SIZE 32

__global__ void matrix_transpose_naive(int *input, int *output)
{

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	int index = indexY * N + indexX;
	int transposedIndex = indexX * N + indexY;

	output[transposedIndex] = input[index];
}

__global__ void matrix_transpose_shared(int *input, int *output)
{
	__shared__ int sharedMemory[BLOCK_SIZE][BLOCK_SIZE];

	// global index	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	// TODO
	// transposed global memory index
	int tindexX = 0;
	int tindexY = 0;

	// TODO
	// local shared memory index
	int localIndexX = 0;
	int localIndexY = 0;

	// Index globaux
	int index = indexY * N + indexX;
	int transposedIndex = tindexY * N + tindexX;

	// TODO remplir le tableau de mémoire partagée à partir de la mémoire global e
	// sharedMemory[..][..] = ...

	// Synchro des thread
	__syncthreads();

	// TODO recopie des données en mémoire globale
	// output[..] = ...
}

//basically just fills the array with random integer between 0 and 99.
void fill_array(int *data) {
	for (int idx = 0; idx < (N*N); idx++)
		data[idx] = rand() % 100;
}

// Check result
void verify_results(int *a, int *b)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			assert(a[i*N + j] == b[j*N + i]);
}

// Print output matrices
void print_output(int *a, int *b) {
	printf("\n Original Matrix::\n");
	for (int idx = 0; idx < (N*N); idx++) {
		if (idx%N == 0)
			printf("\n");
		printf(" %d ", a[idx]);
	}
	printf("\n Transposed Matrix::\n");
	for (int idx = 0; idx < (N*N); idx++) {
		if (idx%N == 0)
			printf("\n");
		printf(" %d ", b[idx]);
	}
}
int main(void) {
	int *a, *b;
	int *d_a, *d_b; // device copies of a, b

	int size = N * N * sizeof(int);

	// Alloc space for host copies of a, b and setup input values
	a = (int *)malloc(size); fill_array(a);
	b = (int *)malloc(size);

	// Alloc space for device copies of a, b
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridSize(N / BLOCK_SIZE, N / BLOCK_SIZE, 1);

	//////////////////////////////
	// Naive version
	//////////////////////////////
	auto t0 = std::chrono::high_resolution_clock::now();

	matrix_transpose_naive << <gridSize, blockSize >> > (d_a, d_b);

	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	auto elapsed_time = std::chrono::duration<double>(t1 - t0).count() * 1000.;
	std::cout << "Naive CUDA transposition: " << elapsed_time << " ms\n";

	// Copy result back to host and check the result
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	verify_results(a, b);

	//////////////////////////////
	// Shared memory version
	//////////////////////////////
	t0 = std::chrono::high_resolution_clock::now();

	matrix_transpose_shared << <gridSize, blockSize >> > (d_a, d_b);

	cudaDeviceSynchronize();
	t1 = std::chrono::high_resolution_clock::now();
	elapsed_time = std::chrono::duration<double>(t1 - t0).count() * 1000.;
	std::cout << "CUDA transposition using shared memory: " << elapsed_time << " ms\n";

	// Copy result back to host and check the result
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	verify_results(a, b);
	// print_output(a,b);

	// terminate memories
	free(a);
	free(b);
	cudaFree(d_a);
	cudaFree(d_b);

	std::cout << "Program complete successfully!" << std::endl;

	return 0;
}
