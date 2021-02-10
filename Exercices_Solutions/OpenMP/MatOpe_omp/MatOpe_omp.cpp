#include <stdio.h>
#include <cmath>
#include <vector>
#include <chrono>

#include <omp.h>

#define NUM_STEPS 100000
#define NB_THROW 5
#define NUM_THREADS 12

//
// Version standard
//
double computePi_std()
{	
	unsigned long long i;
	double x, pi, sum = 0.0;
	double step = 1.0 / (double)NUM_STEPS;

	for (i = 1; i <= NUM_STEPS; i++) {
		x = (i - 0.5)*step;
		sum = sum + 4.0 / (1.0 + x * x);
	}

	pi = step * sum;

	return pi;
}
//
////
//// Version OpenMP
////
//double computePi_omp1()
//{
//	int i;
//	double pi, full_sum = 0.0;
//	double sum[NUM_THREADS];
//
//	double step = 1.0 / (double)NUM_STEPS;
//	omp_set_num_threads(NUM_THREADS);
//
//	#pragma omp parallel
//	{
//		unsigned long long i;
//		int id = omp_get_thread_num();
//		int numthreads = omp_get_num_threads();
//		double x;
//
//		sum[id] = 0.0;
//
//		for (i = id; i < NUM_STEPS; i += numthreads) {
//			x = (i + 0.5)*step;
//			sum[id] = sum[id] + 4.0 / (1.0 + x * x);
//		}
//	}
//
//	for (full_sum = 0.0, i = 0; i < NUM_THREADS; i++)
//		full_sum += sum[i];
//
//	pi = step * full_sum;
//
//	return pi;
//}

double computePi_omp()
{
	int i;
	double pi;
	double sum = 0.;

	double x;

	double step = 1.0 / (double)NUM_STEPS;
	omp_set_num_threads(NUM_THREADS);

#pragma omp parallel  
	{

#pragma omp for reduction(+:sum)
		for (i = 1; i <= NUM_STEPS; i++) {
			x = (i - 0.5)*step;
			sum += 4.0 / (1.0 + x * x);
		}
	}
	pi = step * sum;

	return pi;
}


// Vérification des résultats
void checkRes(const double refValue, const double value)
{
    printf(" - Check results\n");
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
    // Initialisation des données
	double pi = 3.141592653589793;
	double approxPi = -1.;

	std::chrono::high_resolution_clock::time_point start, stop;
	std::chrono::duration<double, std::micro> total_span_std, total_span_omp;

    //
    // Lancement de la version standard
    //
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < NB_THROW; i++)
		approxPi = computePi_std();

	stop = std::chrono::high_resolution_clock::now();
	total_span_std = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    printf("ComputePi (std): %.3fms\n", total_span_std.count() / (double)NB_THROW);

	// Vérification des résultats
	checkRes(pi, approxPi);

    //
    // Lancement de la version OpenMP
    //
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < NB_THROW; i++)
		approxPi = computePi_omp();

	stop = std::chrono::high_resolution_clock::now();
	total_span_omp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    printf("ComputePi (OMP parallel): %.3fms\n", total_span_omp.count() / (double)NB_THROW);
    printf(" - Speedup: %.3f\n", total_span_std / total_span_omp);

    // Vérification des résultats
    checkRes(pi, approxPi);

    return 0;
}

