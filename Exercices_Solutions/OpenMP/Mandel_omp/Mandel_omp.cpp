/***************************************************************************
* Copyright (c) Franck Ginguene											   * 
*                                                                          *
* Bench mutiple implementations of the mandelbrot problem                  *
*  - Scalar																   *
*  - SIMD with xsimd													   *
*  - OpenMP																   *
*  - OpenMP	+ SIMD														   *
*  - CUDA	         													   *
****************************************************************************/

#include <cstdio>
#include <iostream>
#include <string>
#include <omp.h>
#include <fstream>
#include <xsimd-master/examples/pico_bench.hpp>
#include <xsimd-master/include/xsimd/xsimd.hpp>

#define NUM_THREADS 10

// helper function to write the rendered image as PPM file
inline void writePPM(	const std::string &	fileName,
						const int			sizeX,
						const int			sizeY,
						const int *			pixel)
{
	FILE* file = fopen(fileName.c_str(), "wb");
	fprintf(file, "P6\n%i %i\n255\n", sizeX, sizeY);
	unsigned char* out = (unsigned char*)alloca(3 * sizeX);
	for (int y = 0; y < sizeY; y++)
	{
		const unsigned char* in =
			(const unsigned char*)&pixel[(sizeY - 1 - y) * sizeX];

		for (int x = 0; x < sizeX; x++)
		{
			out[3 * x + 0] = in[4 * x + 0];
			out[3 * x + 1] = in[4 * x + 1];
			out[3 * x + 2] = in[4 * x + 2];
		}

		fwrite(out, 3 * sizeX, sizeof(char), file);
	}
	fprintf(file, "\n");
	fclose(file);
}

// omp version ////////////////////////////////////////////////////////////////

namespace omp {

	//#pragma omp declare simd
	template <typename T>
	inline int mandel(T c_re, T c_im, int count)
	{
		T z_re = c_re, z_im = c_im;
		int i;
		for (i = 0; i < count; ++i)
		{
			if (z_re * z_re + z_im * z_im > 4.f)
			{
				break;
			}

			T new_re = z_re * z_re - z_im * z_im;
			T new_im = 2.f * z_re * z_im;
			z_re = c_re + new_re;
			z_im = c_im + new_im;
		}

		return i;
	}

	void mandelbrot(float x0, float y0, float x1, float y1, int width,
		int height, int maxIterations, int output[])
	{
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;
		#pragma omp parallel for num_threads(NUM_THREADS)
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; ++i)
			{
				float x = x0 + i * dx;
				float y = y0 + j * dy;

				//if (omp_get_thread_num() == 1)
				//{
				//	printf("\n");
				//}
				int index = (j * width + i);
				output[index] = omp::mandel<float>(x, y, maxIterations);
			}
		}
	}

} // namespace omp

// scalar version /////////////////////////////////////////////////////////////

namespace scalar {

	inline int mandel(float c_re, float c_im, int count)
	{
		float z_re = c_re, z_im = c_im;
		int i;
		for (i = 0; i < count; ++i)
		{
			if (z_re * z_re + z_im * z_im > 4.f)
			{
				break;
			}

			float new_re = z_re * z_re - z_im * z_im;
			float new_im = 2.f * z_re * z_im;
			z_re = c_re + new_re;
			z_im = c_im + new_im;
		}

		return i;
	}

	void mandelbrot(float x0, float y0, float x1, float y1,
		int width, int height, int maxIterations, int output[])
	{
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; ++i)
			{
				float x = x0 + i * dx;
				float y = y0 + j * dy;

				int index = (j * width + i);
				output[index] = scalar::mandel(x, y, maxIterations);
			}
		}
	}

}  // namespace scalar

int main()
{
	using namespace std::chrono;

	omp_set_num_threads(12);
	const unsigned int width = 450;
	const unsigned int height = 450;
	const float zr = -0.743639266077433f;
	const float zi = +0.131824786875559f;

	const int maxIters = 256;
	constexpr size_t nbiter = 10;
	constexpr int nbpoints = 20;
	float x0_vec, x1_vec, y0_vec, y1_vec;

	float scale = 4.f * (float)std::pow(2., -std::min(13*100 / 60., 53.)*0.7);
	x0_vec = zr - scale;
	x1_vec = zr + scale;
	y0_vec = zi - scale;
	y1_vec = zi + scale;

	alignas(64) std::array<int, width * height> buf;

	auto bencher = pico_bench::Benchmarker<microseconds>{ nbiter, seconds{10} };

	std::cout << "starting benchmarks (results in 'ms')... " << '\n';

	// export CVS
	std::ofstream times_ms_file;
	times_ms_file.open("./mean_times.csv");

	//// scalar run ///////////////////////////////////////////////////////////////
	std::fill(buf.begin(), buf.end(), 0);

	auto stats_scalar = bencher([&]() {
		scalar::mandelbrot(x0_vec, y0_vec, x1_vec, y1_vec, width, height, maxIters, buf.data());
	});
	times_ms_file << stats_scalar.mean().count() << ",";
	std::cout << '\n' << "scalar " << stats_scalar << '\n';

	times_ms_file << "\n";

	// Output (optionel)
	std::string name = "mandelbrot_scalar.ppm";
	writePPM(name, width, height, buf.data());

	//// omp run //////////////////////////////////////////////////////////////////

	std::fill(buf.begin(), buf.end(), 0);

	auto stats_omp = bencher([&]() {
		omp::mandelbrot(x0_vec, y0_vec, x1_vec, y1_vec, width, height, maxIters, buf.data());
	});

	times_ms_file << stats_omp.mean().count() << ",";
	std::cout << '\n' << "omp " << stats_omp << '\n';
	std::cout << " \tmean speedup: " << (float)(stats_scalar.mean().count()) / (float)(stats_omp.mean().count()) << '\n';
	times_ms_file << "\n";

	return 0;
}