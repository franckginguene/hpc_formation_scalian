#include <stdio.h>

#define NB_COLS 8192  // Nombre de colonnes de la matrice.
#define NB_ROWS 8192  // Nombre de lignes de la matrice.
#define NB_THR_X 32   // Nombre de threads par bloc en X.
#define NB_THR_Y 8   // Nombre de threads par bloc en Y.
#define ITER_PER_THR 4

void matrixInit (int *mat);   // Initialisation d'une matrice.
void checkRes (int *mat);     // Vérification des résultats.

#define DIVUP(x,y) (((x)+(y)-1)/(y))
#define IMUL(a,b) __mul24(a,b)

// Noyau CUDA

__global__ void MatrixAdd (int *a, int *b, int *c)
{
    int thrX = blockIdx.x * blockDim.x + threadIdx.x;
    int thrY = blockIdx.y * blockDim.y * ITER_PER_THR + threadIdx.y;
    int index = thrY * NB_COLS + thrX;

    if ((thrX < NB_COLS) && (thrY < NB_ROWS)) {
        c[index] = a[index] + b[index];
        index += NB_COLS * NB_THR_Y;
        c[index] = a[index] + b[index];
        index += NB_COLS * NB_THR_Y;
        c[index] = a[index] + b[index];
        index += NB_COLS * NB_THR_Y;
        c[index] = a[index] + b[index];
    }
}

// Code du Host

int main (void)
{
    int *a, *b, *c;              // Matrices A, B et C du host
    int *dev_a, *dev_b, *dev_c;  // Matrices A, B et C du device

    dim3 threadsPerBlock (NB_THR_X, NB_THR_Y);                                 // Nombre de threads par blocs (en X et en Y).
    dim3 blocksPerGrid (DIVUP(NB_COLS, NB_THR_X), DIVUP(NB_ROWS, (NB_THR_Y * ITER_PER_THR)));   // Nombre de blocs, en X et en Y.

    int matrixSizeB = NB_ROWS * NB_COLS * sizeof(int);

    // Allocation des matrices du host.

    a = (int *)malloc(matrixSizeB);
    if (a == NULL) { printf ("Allocation failure\n"); abort();}

    b = (int *)malloc(matrixSizeB);
    if (b == NULL) { printf ("Allocation failure\n"); abort();}

    c = (int *)malloc(matrixSizeB);
    if (c == NULL) { printf ("Allocation failure\n"); abort();}

    // Allocation des matrices du device.

    cudaMalloc ( (void **) &dev_a, matrixSizeB);
    cudaMalloc ( (void **) &dev_b, matrixSizeB);
    cudaMalloc ( (void **) &dev_c, matrixSizeB);

    // Initialisation des matrices A et B.

    matrixInit(a);
    matrixInit(b);

    // Copie des matrices A et B sur le GPU.

    cudaMemcpy ( dev_a, a, matrixSizeB, cudaMemcpyHostToDevice) ;
    cudaMemcpy ( dev_b, b, matrixSizeB, cudaMemcpyHostToDevice) ;

    // Lancement du noyau.

    MatrixAdd<<<blocksPerGrid, threadsPerBlock>>>( dev_a, dev_b, dev_c );

    // Copie de la matrice C du GPU vers le host.
    cudaMemcpy ( c, dev_c, matrixSizeB, cudaMemcpyDeviceToHost) ;

    checkRes (c);

    // Libération des matrices host et device.

    free (a);
    free (b);
    free (c);

    cudaFree ( dev_a ) ;
    cudaFree ( dev_b ) ;
    cudaFree ( dev_c ) ;

	getchar();

    return 0 ;
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
                printf ("Erreur de calcul sur l'élément %d:%d :\n", l, c);
                printf (" - Valeur calculée : %d\n", mat[l * NB_COLS + c]);
                printf (" - Valeur attendue : %d\n", 2 * (c + l));
                exit(0);
            }

    printf ("NIVEAU 1: Terminé\n");
    printf ("Bravo !\n");
}
