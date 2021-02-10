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
		int id = omp_get_thread_num();

//#pragma omp barrier
#pragma omp single
		{
			std::cout << "OpenMP threads: " << omp_get_num_threads() << std::endl;
		}
		if (omp_get_thread_num() == 2)
		{
			std::cout << ""; 
		}

#pragma omp barrier
		
		printf("Hello from thread %d \n", id);


	}

	return 0;
}