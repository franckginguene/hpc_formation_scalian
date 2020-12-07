#include <stdio.h>
#include <cmath>
#include <vector>

#include <nmmintrin.h> // SSE 4.2
#include <immintrin.h> // AVX

#include <xsimd/xsimd.hpp>

#include "../../tools.h"

#define SIZE (4 * 2048)  // Taille des matrices (matrices carrées)

#define NB_THROW 5

//
// Version standard
//
void matOpeStd(const float* matrixA, const float* matrixB, float* matrixOut, const int nrCols, const int nrRows)
{	
	for (int r = 0; r < nrRows; r++)
	{
		//#pragma loop(no_vector)
		for (int c = 0; c < nrCols; c++)
		{
			matrixOut[r*nrCols + c] = 1.f / sqrt((matrixA[r*nrCols + c] + matrixB[r*nrCols + c]) / sqrt(matrixB[r*nrCols + c]));
		}
	}
}

//
// Version modifiée pour aider le compilateur à réaliser une vectorisation automatique
//
void matOpeAuto(const float* matrixA, const float*  matrixB, float* matrixOut, const int nrCols, const int nrRows)
{
	// TODO !
}

//
// Version SSE
//
void matOpeSSE(const float* __restrict matrixA, const float* __restrict matrixB, float* __restrict matrixOut, const int nrCols, const int nrRows)
{
	// TODO ! Quelques indices :
	// const __m128* __restrict	matA	= (__m128*)matrixA;
	// _mm_div_ps
}

//
// Version AVX
//
void matOpeAVX(const float* __restrict matrixA, const float* __restrict matrixB, float* __restrict matrixOut, const int nrCols, const int nrRows)
{
	// TODO ! Quelques indices :
	// const __m256* __restrict	matA	= (__m256*)matrixA;
	// _mm256_div_ps
}

//
// Version XSIMD
//
void matOpeXSIMD(const float* const __restrict matrixA, float* __restrict matrixB, float* __restrict matrixOut, const int nrCols, const int nrRows)
{
	// TODO ! Pas d'indice, mais voir la page github  ;)
	// https://github.com/xtensor-stack/xsimd
}

//////////////////////////////////////////////
// NE RIEN MODIFIER A PARTIR DE CETTE LIGNE //
//////////////////////////////////////////////

// Vérification des résultats
void checkRes(const float* matrixRef, const float* matrixRes)
{
    printf(" - Check results\n");
	bool resultsOk = true;
    for (int r = 0; r < SIZE && resultsOk; r++)
    {
        for (int c = 0; c < SIZE && resultsOk; c++)
        {
			if (fabs(matrixRes[r * SIZE + c] - matrixRef[r * SIZE + c]) > 1e-3)
            {
                printf ("Error on the element %d:%d\n", r, c);
                printf (" - Computed value: %.6f\n", matrixRes[r * SIZE + c]);
                printf (" - Expected value: %.6f\n", matrixRef[r * SIZE + c]);
				resultsOk = false;
            }
        }
    }
	// Si tout s'est bien passé
	if (resultsOk)
		printf(" - Correct results!\n");
}

//
// Main
//
int main()
{
    float* matrixA;
    float* matrixB;
    float* matrixOutStd;
    float* matrixOutSIMD;

    matrixA			= new float[SIZE * SIZE];
    matrixB			= new float[SIZE * SIZE];
    matrixOutStd	= new float[SIZE * SIZE];
	matrixOutSIMD	= new float[SIZE * SIZE];

    // Initialisation des données

    matrixInit(matrixA, SIZE, SIZE);
    matrixInit(matrixB, SIZE, SIZE);

    mytimer_t start;

    //
    // Lancement de la version standard
    //
    timer_start(&start);

	for (int i = 0; i < NB_THROW; i++)
		matOpeStd(matrixA, matrixB, matrixOutStd, SIZE, SIZE);

    float timStd = timer_stop(&start) / (float)NB_THROW;

    printf("\nMatAdd (std): %.3fs\n", timStd);

    //
    // Lancement de la version vectorisée automatiquement
    //
    timer_start(&start);

	for (int i = 0; i < NB_THROW; i++)
		matOpeAuto(matrixA, matrixB, matrixOutSIMD, SIZE, SIZE);

    float timAuto = timer_stop(&start) / (float)NB_THROW;

    printf("\nMatAdd (vectorized auto): %.3fs\n", timAuto);
    printf(" - Speedup: %.3f\n",       timStd / timAuto);

    // Vérification des résultats
    checkRes(matrixOutStd, matrixOutSIMD);

    //
    // Lancement de la version modifiée 128
    //
    timer_start(&start);

	for (int i = 0; i < NB_THROW; i++)
		matOpeSSE(matrixA, matrixB, matrixOutSIMD, SIZE, SIZE);

    float timSSE = timer_stop(&start) / (float)NB_THROW;

    printf("\nMatAdd (vectorized SSE 128): %.3fs\n", timSSE);
    printf(" - Speedup: %.3f\n",       timStd / timSSE);

    // Vérification des résultats
    checkRes(matrixOutStd, matrixOutSIMD);

	//
	// Lancement de la version modifiée 256
	//
	timer_start(&start);

	for (int i = 0; i < NB_THROW; i++)
		matOpeAVX(matrixA, matrixB, matrixOutSIMD, SIZE, SIZE);

	float timAVX = timer_stop(&start) / (float)NB_THROW;

	printf("\nMatAdd (vectorized AVX 256): %.3fs\n", timAVX);
	printf(" - Speedup: %.3f\n", timStd / timAVX);

	//
	// Vérification des résultats
	//
	checkRes(matrixOutStd, matrixOutSIMD);

	//
	// Lancement de la version xsimd
	//
	timer_start(&start);

	for (int i = 0; i < NB_THROW; i++)
		matOpeXSIMD(matrixA, matrixB, matrixOutSIMD, SIZE, SIZE);

	float timXSIMD = timer_stop(&start) / (float)NB_THROW;

	printf("\nMatAdd (vectorized xsimd): %.3fs\n", timXSIMD);
	printf(" - Speedup: %.3f\n", timStd / timXSIMD);

	//
	// Vérification des résultats
	//
	checkRes(matrixOutStd, matrixOutSIMD);

    return 0;
}

