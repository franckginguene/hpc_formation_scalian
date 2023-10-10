#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 10  // Nombre de donn�es � traiter

// Macro utilitaire pour le retour d'erreur	
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
	// TODO : boucle
}

//
// Code du Host
//
int main (void)
{  
    int dataIn[N];     // Donn�es � traiter (CPU)
    int dataOut[N];    // Donn�es r�sultats (CPU)
    int *dev_dataIn;   // Donn�es � traiter (GPU)
    int *dev_dataOut;  // Donn�es r�sultats (GPU)

    // Initialisation des donn�es
    printf ("Data In:  ");
    for (int i = 0; i < N; i++) {
        dataIn[i] = i;
        printf ("%d ", dataIn[i]);
    }
    printf ("\n");

    // Init timer
    auto t0 = std::chrono::high_resolution_clock::now();

    // Allocation des vecteurs sur le device
    // TODO : allouer les 2 tableaux dev_dataIn et dev_dataOut de taille N
    
    // Copie des donn�es �  traiter sur le GPU.
	// TODO : copier les donn�es de dataIn vers dev_dataIn avec cudaMemcpy

    // Lancement du kernel
    // TODO impl�menter le kernel
    Reverse<<<1, 1>>>( dev_dataIn, dev_dataOut );
    CUT_CHECK_ERROR("Kernel Execution Failed!");

    // Copie des donn�es r�sultats du GPU vers le host.
	// TODO : copie de dev_dataOut vers dataOut

    // Arr�t timer et affichage du temps de calcul
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Execution CUDA : " << elapsed_time << " ms\n";

    // Affichage du r�sultat
    printf ("Data Out: ");
    for (int i = 0; i < N; i++)
        printf ("%d ", dataOut[i]);
    printf ("\n");

    // Lib�ration des vecteurs sur le device
	// TODO lib�ration de la m�moire

    return 0 ;
}

// Exemple de la somme de deux entiers pour vous aider
//int main(void)
//{
//    int c�;
//    int* dev_c�;
//    cudaMalloc((void**)&dev_c, sizeof(int));
//    add << <1, 1 >> > (2, 7, dev_c);
//    cudaMemcpy(&c,
//        dev_c,
//        sizeof(int),
//        cudaMemcpyDeviceToHost);
//    cudaFree(dev_c);
//    return 0;
//}

// Version CPU
//int main(void)
//{
//    int* dataIn = new int[N];     // Donn�es � traiter (CPU)
//    int* dataOut = new int[N];;   // Donn�es r�sultats (CPU)
//
//    // Initialisation des donn�es
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
//    // Arr�t timer et affichage du temps de calcul
//    //cudaDeviceSynchronize();
//    auto t1 = std::chrono::high_resolution_clock::now();
//    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
//    std::cout << "Execution CPU : " << elapsed_time << " ms\n";
//
//    // Affichage du r�sultat
//    printf("Data Out: ");
//    for (int i = 0; i < N; i++)
//        printf("%d ", dataOut[i]);
//    printf("\n");
//
//    // Lib�ration des vecteurs sur l'h�te et le device
//    delete[] dataIn;
//    delete[] dataOut;
//
//    return 0;
//}
