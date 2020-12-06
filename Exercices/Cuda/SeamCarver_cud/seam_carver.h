// Handle error ///////////////////////////////////////////////////////////////

static void HandleError(cudaError_t err,
	const char *file,
	int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// SEAMS //////////////////////////////////////////////////////////////////////

struct seam_link {
	// The X and Y coordinates of the link are inferred by the position of the
	// link in a links array.

	// The minimal energy for any connected seam ending at this position.
	unsigned int energy;

	// The parent X coordinate for vertical seams, Y for horizontal seams.
	int parent_coordinate;
};

struct seam_link * compute_vertical_seam_links(
	const unsigned int *energy,
	int w,
	int h) {
	struct seam_link *links = (seam_link *)malloc(w * h * sizeof(struct seam_link));
	if (!links) {
		fprintf(stderr, "Unable to allocate memory (%d)\n", __LINE__);
		return NULL;
	}

	for (int x = 0; x < w; x++) {
		links[x].energy = energy[x];
		links[x].parent_coordinate = -1;
	}

	for (int y = 1; y < h; y++)
		for (int x = 0; x < w; x++) {
			int i = y * w + x;

			int min_parent_energy = INT_MAX;
			int min_parent_x = -1;

			int parent_x = x == 0 ? x : x - 1;
			int parent_x_end = x == w - 1 ? x : x + 1;
			for (; parent_x <= parent_x_end; parent_x++) {
				int candidate_energy = links[(y - 1) * w + parent_x].energy;
				if (candidate_energy < min_parent_energy) {
					min_parent_energy = candidate_energy;
					min_parent_x = parent_x;
				}
			}

			links[i].energy = energy[i] + min_parent_energy;
			links[i].parent_coordinate = min_parent_x;
		}

	return links;
}

int * get_minimal_seam(
	const struct seam_link *seam_links,
	int num_seams,
	int seam_length) {
	int *minimal_seam = (int *)malloc(seam_length * sizeof(int));
	if (!minimal_seam) {
		fprintf(stderr, "Unable to allocate memory (%d)\n", __LINE__);
	}

	int min_coordinate = -1;
	int min_energy = INT_MAX;

	for (int coordinate = 0; coordinate < num_seams; coordinate++) {
		int i = num_seams * (seam_length - 1) + coordinate;
		if (seam_links[i].energy < min_energy) {
			min_coordinate = coordinate;
			min_energy = seam_links[i].energy;
		}
	}

	int i = 0;
	int offset = min_coordinate;

	for (int d = 0; d < seam_length; d++) {
		minimal_seam[i++] = offset;

		struct seam_link end =
			seam_links[num_seams * (seam_length - 1 - d) + offset];

		offset = end.parent_coordinate;
	}

	return minimal_seam;
}

// REMOVAL ////////////////////////////////////////////////////////////////////

unsigned char * image_after_vertical_seam_removal(
	const unsigned char *original_data,
	const int *vertical_seam,
	int w,
	int h) {
	unsigned char *img = (unsigned char *)malloc((w - 1) * h * 3);
	if (!img) {
		fprintf(stderr, "Unable to allocate memory (%d)\n", __LINE__);
		return NULL;
	}

	for (int y = 0; y < h; y++) {
		int seamx = vertical_seam[h - 1 - y];

		for (int x = 0, imgx = 0; imgx < w - 1; x++, imgx++) {
			if (x == seamx) { x++; }

			int    i = (y *  w + x) * 3;
			int imgi = (y * (w - 1) + imgx) * 3;

			img[imgi] = original_data[i];
			img[imgi + 1] = original_data[i + 1];
			img[imgi + 2] = original_data[i + 2];
		}
	}

	return img;
}

// OUTPUT /////////////////////////////////////////////////////////////////////

int write_energy(
	const unsigned int *energy,
	int w,
	int h,
	const char *filename) {
	int result = 0;

	unsigned char *energy_normalized = (unsigned char *)malloc(w * h);
	if (!energy_normalized) {
		fprintf(stderr, "Unable to allocate memory (%d)\n", __LINE__);

		result = 1;
		//goto cleanup;
	}

	int max_energy = 1;
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			int i = y * w + x;
			max_energy = energy[i] > max_energy ? energy[i] : max_energy;
		}

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			int i = y * w + x;
			energy_normalized[i] = (char)((double)energy[i] / max_energy * 255);
		}

	printf("Writing to '%s'\n", filename);
	if (!stbi_write_jpg(filename, w, h, 1, energy_normalized, 80)) {
		fprintf(stderr, "Unable to write output (%d)\n", __LINE__);

		result = 1;
	}

	return result;
}

int draw_vertical_seam(
	const unsigned char *data,
	const int *minimal_vertical_seam,
	int w,
	int h,
	const char *filename) {
	int result = 0;

	unsigned char *data_with_seams = (unsigned char *)malloc(w * h * 3);
	if (!data_with_seams) {
		fprintf(stderr, "Unable to allocate memory (%d)\n", __LINE__);

		result = 1;
	}

	memcpy(data_with_seams, data, w * h * 3);

	for (int y = h - 1; y >= 0; y--) {
		int x = minimal_vertical_seam[h - 1 - y];
		int i = (y * w + x) * 3;

		data_with_seams[i] = 255;
		data_with_seams[i + 1] = 0;
		data_with_seams[i + 2] = 0;
	}

	printf("Writing to '%s'\n", filename);
	if (!stbi_write_jpg(filename, w, h, 3, data_with_seams, 80)) {
		fprintf(stderr, "Unable to write output (%d)\n", __LINE__);

		result = 1;
	}

	return result;
}

int draw_image(
	const unsigned char *data,
	int w,
	int h,
	const char *filename) {
	printf("Writing %dx%d image to '%s'\n", w, h, filename);
	return stbi_write_jpg(filename, w, h, 3, data, 80);
}
