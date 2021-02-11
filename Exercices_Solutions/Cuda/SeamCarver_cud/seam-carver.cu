#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "seam_carver.h"

// ENERGY /////////////////////////////////////////////////////////////////////

unsigned int energy_at_cpu(const unsigned char *data,int w,int h,int x,int y) 
{
	int x0 = x == 0 ? x : x - 1;
	int x1 = x == w - 1 ? x : x + 1;
	int ix0 = (y * w + x0) * 3;
	int ix1 = (y * w + x1) * 3;
	unsigned int dxr = data[ix0    ] - data[ix1    ];
	unsigned int dxg = data[ix0 + 1] - data[ix1 + 1];
	unsigned int dxb = data[ix0 + 2] - data[ix1 + 2];
	unsigned int dx = dxr * dxr + dxg * dxg + dxb * dxb;

	int y0 = y == 0 ? y : y - 1;
	int y1 = y == h - 1 ? y : y + 1;
	int iy0 = (y0 * w + x) * 3;
	int iy1 = (y1 * w + x) * 3;
	unsigned int dyr = data[iy0    ] - data[iy1    ];
	unsigned int dyg = data[iy0 + 1] - data[iy1 + 1];
	unsigned int dyb = data[iy0 + 2] - data[iy1 + 2];
	unsigned int dy = dyr * dyr + dyg * dyg + dyb * dyb;

	return dx + dy;
}

unsigned int * compute_energy_cpu(const unsigned char *data, int w, int h) {
unsigned int *energy = (unsigned int *)malloc(w * h * sizeof(unsigned int));
if (!energy) {
    fprintf(stderr, "Unable to allocate memory (%d)\n", __LINE__);
    return NULL;
}

for (int y = 0; y < h; y++)
{
    for (int x = 0; x < w; x++) 
    {
        int i = y * w + x;
        energy[i] = energy_at_cpu(data, w, h, x, y);
    }
}

return energy;
}

//compute energy with GPU version
__global__ void compute_energy_kernel(unsigned char * data_GPU, unsigned int * energy_GPU, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < w && y < h)
    {
        int x0 = x == 0 ? x : x - 1;
        int x1 = x == w - 1 ? x : x + 1;
        int ix0 = (y * w + x0) * 3;
        int ix1 = (y * w + x1) * 3;
        unsigned int dxr = data_GPU[ix0    ] - data_GPU[ix1    ];
        unsigned int dxg = data_GPU[ix0 + 1] - data_GPU[ix1 + 1];
        unsigned int dxb = data_GPU[ix0 + 2] - data_GPU[ix1 + 2];
        unsigned int dx = dxr * dxr + dxg * dxg + dxb * dxb;
    
        int y0 = y == 0 ? y : y - 1;
        int y1 = y == h - 1 ? y : y + 1;
        int iy0 = (y0 * w + x) * 3;
        int iy1 = (y1 * w + x) * 3;
        unsigned int dyr = data_GPU[iy0    ] - data_GPU[iy1    ];
        unsigned int dyg = data_GPU[iy0 + 1] - data_GPU[iy1 + 1];
        unsigned int dyb = data_GPU[iy0 + 2] - data_GPU[iy1 + 2];
        unsigned int dy = dyr * dyr + dyg * dyg + dyb * dyb;
        
        energy_GPU[y * w + x] = dx + dy;
    }
}

unsigned int * compute_energy(const unsigned char *data, int w, int h) {
    // grid and block size
    dim3 block_size(16,16);
    dim3 grid_size((unsigned int)ceil((float)w / block_size.x), (unsigned int)ceil((float)h / block_size.y));

    // Allocation
    int size = w * h;
    unsigned char * data_GPU = nullptr;
    unsigned int * energy_GPU = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&data_GPU, size * 3 * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMalloc((void**)&energy_GPU, size * sizeof(unsigned int)));

    // Copy
    HANDLE_ERROR(cudaMemcpy(data_GPU, data, sizeof(unsigned char) * 3 * size, cudaMemcpyHostToDevice));

    // Call kernel
    compute_energy_kernel<<<grid_size, block_size>>>(data_GPU, energy_GPU, w, h);

    // Get energy map back to CPU
    unsigned int *energy = (unsigned int *)malloc(w * h * sizeof(unsigned int));
    if (!energy) {
        fprintf(stderr, "Unable to allocate memory (%d)\n", __LINE__);
        return NULL;
    }
    HANDLE_ERROR(cudaMemcpy(energy, energy_GPU, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(data_GPU));
    HANDLE_ERROR(cudaFree(energy_GPU));
    
    return energy;
}

// MAIN ///////////////////////////////////////////////////////////////////////

void show_usage(const char *program) {
    fprintf(
            stderr,
            "USAGE:\n"
            "  %s <input-filename> <output-directory> <num-iterations>\n",
            program);
}

unsigned char * run_iteration(const char *output_directory, const unsigned char *data, int w, int h, int iteration, double & elapsed_time)
{
    unsigned int *	energy = NULL;
    struct			seam_link *vertical_seam_links = NULL;
    int *			minimal_vertical_seam = NULL;
    unsigned char *	output_data = NULL;

    char output_filename[1024];

	auto t0 = std::chrono::high_resolution_clock::now();

    energy = compute_energy(data, w, h);

	auto t1 = std::chrono::high_resolution_clock::now();
	elapsed_time = std::chrono::duration<double>(t1 - t0).count();

    if (iteration == 0) {
        snprintf(output_filename , 1024, "%s/img-energy%d.jpg", output_directory, iteration);
        if (write_energy(energy, w, h, output_filename)) {
        }
    }

    vertical_seam_links = compute_vertical_seam_links(energy, w, h);
    free(energy);
    if (!vertical_seam_links) { }

    minimal_vertical_seam = get_minimal_seam(vertical_seam_links, w, h);
    free(vertical_seam_links);

    snprintf(
            output_filename,
            1024,
            "%s/img-seam-%04d.jpg",
            output_directory,
            iteration);
    printf("Iteration: %d\n", iteration);
    // write output image 
    // if (draw_vertical_seam(
    //             data,
    //             minimal_vertical_seam,
    //             w,
    //             h,
    //             output_filename)) {
    // }

    output_data =
        image_after_vertical_seam_removal(data, minimal_vertical_seam, w, h);

    return output_data;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        show_usage(argv[0]);
        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_directory = argv[2];
    int num_iterations = atoi(argv[3]);

    int result = 0;

    unsigned char *initial_img = NULL;
    unsigned char *data = NULL;

    printf("Reading '%s'\n", input_filename);

    int w, h, n;
    initial_img = stbi_load(input_filename, &w, &h, &n, 3);
    if (!initial_img) {
        fprintf(stderr, "Unable to read '%s'\n", input_filename);

        result = 1;
    }

    printf("Loaded %dx%d image\n", w, h);

    data = initial_img;
	double total_elapsed_time = 0.;
    for (int i = 0; i < num_iterations; i++) {
		double elapsed_time = 0.;
        unsigned char *next_data = run_iteration(output_directory, data, w, h, i, elapsed_time);
		total_elapsed_time += elapsed_time;
        if (!next_data) {
            fprintf(stderr, "Error running iteration %d\n", i);

            result = 1;
        }

        if (i > 0) { free(data); }
        data = next_data;
        w--;
    }
    std::cout << "Seam carving energy image compute in " << total_elapsed_time << " seconds\n";

    char resized_output_filename[1024];
    snprintf(resized_output_filename, 1024, "%s/img.jpg", output_directory);
    if (!draw_image(data, w, h, resized_output_filename)) 
	{
        fprintf(stderr, "\033[1;31mUnable to write %s\033[0m\n", resized_output_filename);
    }

    return result;
}
