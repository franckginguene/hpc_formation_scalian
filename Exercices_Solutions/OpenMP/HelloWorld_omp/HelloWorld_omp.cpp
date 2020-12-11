#include <iostream>
#include <stdio.h>

#include <omp.h>

//
// Main
//
int main()
{
	// Sortie du thread master
	std::cout << "Hello from master thread" << "\n";

	// Nombre de threads
	std::cout << "OpenMP threads: " << omp_get_num_threads() << "\n";

	// Paramétrer le nombre de threads
	omp_set_num_threads(4);

	// Région parallèle
#pragma omp parallel
	{
#pragma omp single
		std::cout << "OpenMP threads: " << omp_get_num_threads() << std::endl;
		printf("Hello from thread %d \n", omp_get_thread_num());
	}

	return 0;
}
