#include <stdio.h>

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
	// TODO : boucle for
}

//
// Code du Host
//
int main (void)
{  
    int dataIn[N];     // Données à traiter (CPU)
    int dataOut[N];    // Données résultats (CPU)
    int *dev_dataIn;   // Données à traiter (GPU)
    int *dev_dataOut;  // Données résultats (GPU)

    // Allocation des vecteurs sur le device
    // TODO : allouer les 2 tableaux dev_dataIn et dev_dataOut de taille N

    // Initialisation des données
    printf ("Data In:  ");
    for (int i = 0; i < N; i++) {
        dataIn[i] = i;
        printf ("%d ", dataIn[i]);
    }
    printf ("\n");

    // Copie des données à  traiter sur le GPU.
	// TODO : copier les données de dataIn vers dev_dataIn avec cudaMemcpy

    // Lancement du noyau.
    Reverse<<<1, 1>>>( dev_dataIn, dev_dataOut );
    CUT_CHECK_ERROR("Kernel Execution Failed!");

    // Copie des données résultats du GPU vers le host.
	// TODO : copie de dev_dataOut vers dataOut

    // Affichage du résultat
    printf ("Data Out: ");
    for (int i = 0; i < N; i++)
        printf ("%d ", dataOut[i]);
    printf ("\n");

    // Libération des vecteurs sur le device
	// TODO libération de la mémoire

    return 0 ;
}
