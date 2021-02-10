// This program shows are the basics of using the parallel STL in C++
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

// Helper function for getting the current time for profiling
auto get_time() { return std::chrono::high_resolution_clock::now(); }

int main() {
  // Size of our vector
  const int N = 1 << 20;

  // Create our vector
  std::vector<int> v(N);

  // Set up our random number generator
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<int> dist(0, 255);

  // Generate our random inputs
  std::generate(begin(v), end(v), [&]() { return dist(rng); });

  // Reduce the vector the classic way
  auto start = get_time();
  std::sort(begin(v), end(v));
  auto finish = get_time();

  // Print out the execution time
  auto duration =
	  std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  std::cout << "Elapsed time std = " << duration.count() << " ms\n";

  // Init and reduce the vector in parallel and vectorized
  std::generate(begin(v), end(v), [&]() { return dist(rng); });
  start = get_time();
  std::sort(std::execution::par, begin(v), end(v));
  finish = get_time();

  // Print out the execution time
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  std::cout << "Elapsed time parallel = " << duration.count() << " ms\n";

  return 0;
}