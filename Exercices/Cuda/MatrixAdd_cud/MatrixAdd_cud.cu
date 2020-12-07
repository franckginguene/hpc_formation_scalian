#include <stdio.h>

#define NB_COLS 4000  // Nombre de colonnes de la matrice.
#define NB_ROWS	4000  // Nombre de lignes de la matrice.

void matrixInit(int *mat);   // Initialisation d'une matrice.
void checkRes(int *mat);     // Vérification des résultats.

// Noyau CUDA
__global__ void MatrixAdd(int *a, int *b, int *c)
{
	// TODO
	// indices : calculez l'indice global à l'aide de 
	// blockIdx.x
	// blockDim.x
	// threadIdx.x
}

// Code du Host
int main(void)
{
	int *a, *b, *c;              // Matrices A, B et C du host
	int *dev_a, *dev_b, *dev_c;  // Matrices A, B et C du device

	int nbElements = NB_COLS * NB_ROWS;
	int matrixSize = nbElements * sizeof(int);
	int threadsPerBlock = 256;
	// TODO calculez le nombre de blocs nécessaires
	int blocksPerGrid = 0;

	// Allocation des matrices du host.
	a = (int *)malloc(matrixSize);
	if (a == NULL) { printf("Allocation failure\n"); abort(); }

	b = (int *)malloc(matrixSize);
	if (b == NULL) { printf("Allocation failure\n"); abort(); }

	c = (int *)malloc(matrixSize);
	if (c == NULL) { printf("Allocation failure\n"); abort(); }

	// Allocation des matrices du device.
	cudaMalloc((void **)&dev_a, matrixSize);
	cudaMalloc((void **)&dev_b, matrixSize);
	cudaMalloc((void **)&dev_c, matrixSize);
	auto cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel memory failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// Initialisation des matrices A et B.
	matrixInit(a);
	matrixInit(b);

	// Copie des matrices A et B sur le GPU.
	// TODO transférer les données a et b vers le GPU
	// utilisez cudaMemcpy et cudaMemcpyHostToDevice

	// Lancement du noyau.
	// TODO lancer le kernel avec le nombre de bloc et le nombre de threads par bloc adéquats
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

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

	printf("LEVEL 1: Done\n");
	printf("Good job!\n");
}
