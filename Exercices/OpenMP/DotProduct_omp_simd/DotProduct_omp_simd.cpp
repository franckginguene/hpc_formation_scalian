#include <algorithm>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <chrono>
#include <omp.h>

#define SIZE (2*2*2*2*3*3*5*7*11*13)  // Taille des vecteurs : ppcm des nombres entre 1 et 13
#define NUM_THREADS 8
#define NB_THROW 500

//
// Version séquentielle
//
double dotProductSeq(const std::vector<float> & vectorA, const std::vector<float> & vectorB)
{
    
	double res = 0;

    for (int i = 0; i < SIZE; i++)
    {
        res += vectorA[i] * vectorB[i];
    }

    return res;
}

//
// Version OpenMP parallel
//
double dotProductParallel(const std::vector<float> & vectorA, const std::vector<float> & vectorB)
{

	double res = 0;

#pragma omp parallel for
	for (int i = 0; i < SIZE; i++)
	{
//#pragma omp simd
		res += vectorA[i] * vectorB[i];
	}

	return res;
}

//
// Version OpenMP sans reduction
//
double dotProductParallelFor(const std::vector<float> & vectorA, const std::vector<float> & vectorB)
{
	// TODO
	// indices : 
	//		#pragma omp for
	double res = 0;
	return res;
}

//
// Version OpenMP avec reduction
//
double dotProductRedux(const std::vector<float> & vectorA, const std::vector<float> & vectorB)
{
	// TODO
	// indices : 
	//		#pragma omp parallel for reduction(+:res)
    double res = 0;
    return res;
}

//
// Version OpenMP awful version
//
double dotProductawful(const std::vector<float> & vectorA, const std::vector<float> & vectorB)
{
	double res = 0;

	#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < SIZE; i++)
	{
		#pragma omp atomic
		res += vectorA[i] * vectorB[i];
	}

	return res;
}

// Vérification des résultats
void checkRes(const double refValue, const double value)
{
	bool resultsOk = fabs(refValue - value) < 1.e-9;

	// Si tout s'est bien passé
	if (resultsOk)
	{
		printf(" - Correct results!\n");
	}
	else
	{
		printf(" - Wrong results!\n");
		printf("The value is %.15f\n", value);
		printf("The expected value is %.15f\n", refValue);
	}
}

//
// Main
//
int main()
{
	double    ref_res, current_res;
    std::vector<float>    vectorA(SIZE);
	std::vector<float>    vectorB(SIZE);
	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::duration<double, std::micro> total_time_std, total_time_omp;

	// Initialisation des données
	std::generate(vectorA.begin(), vectorA.end(), [n = 0.]() mutable { return n + 1e-5; });
	std::iota(vectorB.begin(), vectorB.end(), -SIZE / 2);

    //
    // Version séquentielle
    //
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < NB_THROW; i++)
		ref_res = dotProductSeq(vectorA, vectorB);

	stop = std::chrono::high_resolution_clock::now();
	total_time_std = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	printf("Dot product (std): %.3f microsec\n", total_time_std.count() / (double)NB_THROW);

	//
	// Version OpenMP parallel
	//
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < NB_THROW; i++)
		current_res = dotProductParallel(vectorA, vectorB);

	stop = std::chrono::high_resolution_clock::now();
	total_time_omp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	printf("\nDot product (OMP parallel): %.3f microsec\n", total_time_omp.count() / (double)NB_THROW);
	printf(" - Speedup: %.3f\n", total_time_std / total_time_omp);

	// Vérification des résultats
	checkRes(ref_res, current_res);

	//
	// Version OpenMP parallel for
	//
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < NB_THROW; i++)
		current_res = dotProductParallelFor(vectorA, vectorB);

	stop = std::chrono::high_resolution_clock::now();
	total_time_omp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	printf("\nDot product (OMP parallel for atomic): %.3f microsec\n", total_time_omp.count() / (double)NB_THROW);
	printf(" - Speedup: %.3f\n", total_time_std / total_time_omp);

	// Vérification des résultats
	checkRes(ref_res, current_res);

    //
    // Version OpenMP avec réduction
    //
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < NB_THROW; i++)
		current_res = dotProductRedux(vectorA, vectorB);

	stop = std::chrono::high_resolution_clock::now();
	total_time_omp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	printf("\nDot product (OMP parallel for redux): %.3f microsec\n", total_time_omp.count() / (double)NB_THROW);
	printf(" - Speedup: %.3f\n", total_time_std / total_time_omp);

	// Vérification des résultats
	checkRes(ref_res, current_res);

	//
	// Version OpenMP à ne pas faire...
	//
	//start = std::chrono::high_resolution_clock::now();

	//for (int i = 0; i < NB_THROW; i++)
	//	current_res = dotProductawful(vectorA, vectorB);

	//stop = std::chrono::high_resolution_clock::now();
	//total_time_omp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	//printf("\nDot product (OMP awful): %.3f microsec\n", total_time_omp.count() / (double)NB_THROW);
	//printf(" - Speedup: %.3f\n", total_time_std / total_time_omp);

	//// Vérification des résultats
	//checkRes(ref_res, current_res);

    return 0;
}
