#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 10  // Nombre de données à traiter

// Macro utilitaire de retour d'erreur	
#define CUT_CHECK_ERROR(errorMessage) {											\
	  cudaError_t err = cudaGetLastError();										\
	  if( cudaSuccess != err) {													\
		  fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",		\
				  errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );	\
		  exit(EXIT_FAILURE);													\
	  }																			\
  }

//
// Noyau CUDA
//
__global__ void Reverse (int *dataIn, int *dataOut)
{
    for (int i = 0; i < N; i++)
        dataOut[i] = dataIn[(N - 1) - i];
}

//
// Noyau CUDA, 2eme version
//
__global__ void Reverse2(int *dataIn, int *dataOut)
{
	int id = threadIdx.x;
	dataOut[id] = dataIn[(N - 1) - id];
}

//
// Code du Host
//
int main(void)
{
    int* dataIn = new int[N];     // Données à traiter (CPU)
    int* dataOut = new int[N];;   // Données résultats (CPU)
    int* dev_dataIn;              // Données à traiter (GPU)
    int* dev_dataOut;             // Données résultats (GPU)

    // Initialisation des données
    printf("Data In:  ");
    for (int i = 0; i < N; i++) {
        dataIn[i] = i;
        printf("%d ", dataIn[i]);
    }
    printf("\n");

    // Init timer
    auto t0 = std::chrono::high_resolution_clock::now();

    // Allocation des vecteurs sur le device
    cudaMalloc((void**)&dev_dataIn, N * sizeof(int));
    CUT_CHECK_ERROR("Memory allocation In Failed!");
    cudaMalloc((void**)&dev_dataOut, N * sizeof(int));
    CUT_CHECK_ERROR("Memory allocation Out Failed!");

    // Copie des données à  traiter sur le GPU.
    cudaMemcpy(dev_dataIn, dataIn, N * sizeof(int), cudaMemcpyHostToDevice);
    CUT_CHECK_ERROR("Memory copy Failed!");
    // Lancement du noyau.
    Reverse<<<1, 1>>>( dev_dataIn, dev_dataOut );
    // Reverse2 << < 1, N >> > (dev_dataIn, dev_dataOut);
    CUT_CHECK_ERROR("Kernel Execution Failed!");

    // Copie des données résultats du GPU vers le host.
    cudaMemcpy(dataOut, dev_dataOut, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Arrêt timer et affichage du temps de calcul
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Execution CUDA : " << elapsed_time << " ms\n";

    // Affichage du résultat
    printf("Data Out: ");
    for (int i = 0; i < N; i++)
        printf("%d ", dataOut[i]);
    printf("\n");

    // Libération des vecteurs sur l'hôte et le device
    cudaFree(dev_dataIn);
    cudaFree(dev_dataOut);
    delete[] dataIn;
    delete[] dataOut;

    return 0;
}

// Version CPU
//int main(void)
//{
//    int* dataIn = new int[N];     // Données à traiter (CPU)
//    int* dataOut = new int[N];;   // Données résultats (CPU)
//
//    // Initialisation des données
//    printf("Data In:  ");
//    for (int i = 0; i < N; i++) {
//        dataIn[i] = i;
//        //printf("%d ", dataIn[i]);
//    }
//    printf("\n");
//
//    // Init timer
//    auto t0 = std::chrono::high_resolution_clock::now();
//
//    // Lancement du calcul
//    for (int i = 0; i < N; i++)
//        dataOut[i] = dataIn[(N - 1) - i];
//
//    // Arrêt timer et affichage du temps de calcul
//    auto t1 = std::chrono::high_resolution_clock::now();
//    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
//    std::cout << "Execution CPU : " << elapsed_time << " ms\n";
//
//    // Affichage du résultat
//    printf("Data Out: ");
//    for (int i = 0; i < N; i++)
//        printf("%d ", dataOut[i]);
//    printf("\n");
//
//    // Libération des vecteurs sur l'hôte et le device
//    delete[] dataIn;
//    delete[] dataOut;
//
//    return 0;
//}