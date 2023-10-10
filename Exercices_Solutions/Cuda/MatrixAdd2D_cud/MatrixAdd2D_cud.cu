#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#define NB_COLS 1000  // Nombre de colonnes de la matrice.
#define NB_ROWS	500  // Nombre de lignes de la matrice.
#define NB_THREADS 16 // Nombre de threads par bloc dans 1 dimension

void matrixInit(int *mat);   // Initialisation d'une matrice.
void checkRes(int *mat);     // Vérification des résultats.

// Noyau CUDA
__global__ void MatrixAdd(int *a, int *b, int *c)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < NB_COLS && y < NB_ROWS)
	{
		int globalId = y * NB_COLS + x;
		c[globalId] = a[globalId] + b[globalId];
	}
}

// Code du Host
int main(void)
{
	int *a, *b, *c;              // Matrices A, B et C du host
	int *dev_a, *dev_b, *dev_c;  // Matrices A, B et C du device

	int nbElements = NB_COLS * NB_ROWS;
	int matrixSize = nbElements * sizeof(int);
	dim3 threadsPerBlock(NB_THREADS, NB_THREADS);
	dim3 blocksPerGrid ((NB_COLS + NB_THREADS - 1) / NB_THREADS, (NB_ROWS + NB_THREADS - 1) / NB_THREADS);

	std::cout << threadsPerBlock.x << '\t' << threadsPerBlock.y << '\t' << threadsPerBlock.z << std::endl;
	std::cout << blocksPerGrid.x << '\t' << blocksPerGrid.y << '\t' << blocksPerGrid.z << std::endl;

	// Allocation des matrices du host.
	a = (int *)malloc(matrixSize);
	if (a == NULL) { printf("Allocation failure\n"); abort(); }

	b = (int *)malloc(matrixSize);
	if (b == NULL) { printf("Allocation failure\n"); abort(); }

	c = (int *)malloc(matrixSize);
	if (c == NULL) { printf("Allocation failure\n"); abort(); }

	// Initialisation des matrices A et B.
	matrixInit(a);
	matrixInit(b);

	// Allocation des matrices du device.
	cudaMalloc((void **)&dev_a, matrixSize);
	cudaMalloc((void **)&dev_b, matrixSize);
	cudaMalloc((void **)&dev_c, matrixSize);
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel memory failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// Copie des matrices A et B sur le GPU.
	cudaMemcpy(dev_a, a, matrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, matrixSize, cudaMemcpyHostToDevice);


	// Init timer
	cudaDeviceSynchronize();
	auto t0 = std::chrono::high_resolution_clock::now();

	// Lancement du noyau.
	MatrixAdd <<<blocksPerGrid, threadsPerBlock>>> (dev_a, dev_b, dev_c);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// Arrêt timer et affichage du temps de calcul
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	std::cout << "Execution CUDA : " << elapsed_time << " micro seconds\n";

	// Copie de la matrice C du GPU vers le host.
	cudaMemcpy(c, dev_c, matrixSize, cudaMemcpyDeviceToHost);

	checkRes(c);

	// Libération des matrices host et device.
	free(a);
	free(b);
	free(c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	getchar();

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//
// Fonctions outils. Rien à modifier.
//
/////////////////////////////////////////////////////////////////////////////////////////////

// Version CPU uniquement
//int main(void)
//{
//	int* a, * b, * c;              // Matrices A, B et C du host
//
//	int nbElements = NB_COLS * NB_ROWS;
//	int matrixSize = nbElements * sizeof(int);
//
//	// Allocation des matrices du host.
//	a = (int*)malloc(matrixSize);
//	if (a == NULL) { printf("Allocation failure\n"); abort(); }
//
//	b = (int*)malloc(matrixSize);
//	if (b == NULL) { printf("Allocation failure\n"); abort(); }
//
//	c = (int*)malloc(matrixSize);
//	if (c == NULL) { printf("Allocation failure\n"); abort(); }
//
//	// Initialisation des matrices A et B.
//	matrixInit(a);
//	matrixInit(b);
//
//	// Init timer
//	auto t0 = std::chrono::high_resolution_clock::now();
//
//	for (int i = 0; i < NB_ROWS; i++)
//		for (int j = 0; j < NB_COLS; j++)
//			c[i * NB_COLS + j] = a[i * NB_COLS + j] + b[i * NB_COLS + j];
//
//	// Arrêt timer et affichage du temps de calcul
//	auto t1 = std::chrono::high_resolution_clock::now();
//	auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
//	std::cout << "Execution CPU : " << elapsed_time << " micro seconds\n";
//
//	checkRes(c);
//
//	// Libération des matrices host et device.
//	free(a);
//	free(b);
//	free(c);
//
//	getchar();
//
//	return 0;
//}

void matrixInit(int *mat)
{
	int l, c;

	for (l = 0; l < NB_ROWS; l++)
		for (c = 0; c < NB_COLS; c++)
			mat[l * NB_COLS + c] = l + c;
}

void checkRes(int *mat)
{
	int l, c;

	for (l = 0; l < NB_ROWS; l++)
		for (c = 0; c < NB_COLS; c++)
			if (mat[l * NB_COLS + c] != 2 * (c + l)) {
				printf("Erreur de calcul sur l'element %d:%d :\n", l, c);
				printf(" - Valeur calculee : %d\n", mat[l * NB_COLS + c]);
				printf(" - Valeur attendue : %d\n", 2 * (c + l));
				exit(0);
			}

	printf("LEVEL 2: Done\n");
	printf("Good job!\n");
}
