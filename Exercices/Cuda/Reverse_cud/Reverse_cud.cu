#include <stdio.h>

#define N 10  // Nombre de donn�es � traiter

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
	// TODO : boucle for
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

    // Allocation des vecteurs sur le device
    // TODO : allouer les 2 tableaux dev_dataIn et dev_dataOut de taille N

    // Initialisation des donn�es
    printf ("Data In:  ");
    for (int i = 0; i < N; i++) {
        dataIn[i] = i;
        printf ("%d ", dataIn[i]);
    }
    printf ("\n");

    // Copie des donn�es �  traiter sur le GPU.
	// TODO : copier les donn�es de dataIn vers dev_dataIn avec cudaMemcpy

    // Lancement du noyau.
    Reverse<<<1, 1>>>( dev_dataIn, dev_dataOut );
    CUT_CHECK_ERROR("Kernel Execution Failed!");

    // Copie des donn�es r�sultats du GPU vers le host.
	// TODO : copie de dev_dataOut vers dataOut

    // Affichage du r�sultat
    printf ("Data Out: ");
    for (int i = 0; i < N; i++)
        printf ("%d ", dataOut[i]);
    printf ("\n");

    // Lib�ration des vecteurs sur le device
	// TODO lib�ration de la m�moire

    return 0 ;
}
