#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#define __restrict __restrict__
#endif

///////////////////////////////////////////////////////////////////////////
//
// Fonctions de mesure du temps
//
///////////////////////////////////////////////////////////////////////////

#ifdef WIN32
typedef clock_t mytimer_t;
static inline void timer_start(clock_t *start)
{
    *start  = clock();
}

static inline float timer_stop(clock_t *start)
{
    clock_t stop = clock();

    return (float)(stop - *start) / CLOCKS_PER_SEC;
}
#else
typedef struct timeval mytimer_t;
static inline void timer_start(struct timeval * tv)
{
    gettimeofday(tv, NULL);
}

static inline float timer_stop(struct timeval * start)
{
    struct timeval stop;

    gettimeofday(&stop, NULL);
    return ((float)(stop.tv_sec - start->tv_sec) + 1e-6*(float)(stop.tv_usec -
                                                                start->tv_usec));
}
#endif

///////////////////////////////////////////////////////////////////////////
//
// Cuda tools
//
///////////////////////////////////////////////////////////////////////////

#define CUDA_SAFE_CALL(call)                                                                                        \
 do                                                                                                                 \
 {                                                                                                                  \
  cudaError_t err = call;                                                                                           \
  if(cudaSuccess != err)                                                                                            \
  {                                                                                                   \
    fprintf(stderr, "%s: CUDA Error: (%d) %s\n", __PRETTY_FUNCTION__, err, cudaGetErrorString(err));  \
    fflush(stderr);                                                                                                 \
  }                                                                                                                 \
}while (0)

#define CUT_CHECK_ERROR(errorMessage) {                                      \
  cudaError_t err = cudaGetLastError();                                    \
  if( cudaSuccess != err) {                                                \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
      exit(EXIT_FAILURE);                                                  \
  }                                                                        \
  }

#define DIVUP(x,y) (((x)+(y)-1)/(y))
#define IMUL(a,b) __mul24(a,b)

static inline void kernelInit(float *mat, int kernel_size)
{
    int l, c;

    for (l = 0; l < kernel_size; l++)
        for (c = 0; c < kernel_size; c++)
            mat[l * kernel_size + c] = (float)(l + c) / (kernel_size * kernel_size);
}

void checkConvolutionRes(float *matIn, float *matOut, float *kernel, int nb_cols, int nb_rows, int border, int kernel_size)
{
    int l, c, x, y;
    int kernelIndex;
    int kernel_radius = kernel_size / 2;
    int nb_cols_border = nb_cols + 2 * border;
    double res, err;

    printf ("Check Convolution results: ");
    fflush (stdout);

    if (border == 0)
        abort();

    for (l = 0; l < nb_rows; l++)
        for (c = 0; c < nb_cols; c++) {
            res = 0;
            kernelIndex = 0;
            for (y = l + border - kernel_radius; y <= l +border + kernel_radius; y++) {
                for (x = c +border - kernel_radius; x <= c + border + kernel_radius; x++) {
                    res += matIn[y * nb_cols_border + x] * kernel[kernelIndex];
                    kernelIndex++;
                }
            }
            err =fabs((matOut[l * nb_cols + c] - res) / res);
            if (err > 0.000001) {
                printf ("Erreur at %d;%d : %f / %f - Error : %f\n", l, c, matOut[l * nb_cols + c], res, err);
                abort();
            }
        }

    printf ("Success !\n");
}

static inline void matrixInitB(float *mat, int nb_cols, int nb_rows, int border)
{
    int nb_cols_border = nb_cols + 2 * border;
    int nb_rows_border = nb_rows + 2 * border;
    int matrixSizeBorderB = nb_cols_border * nb_rows_border * sizeof(float);
    int l, c;

    memset (mat, 0, matrixSizeBorderB);

    for (l = border; l < nb_rows + border; l++)
        for (c = border; c < nb_cols + border; c++)
            mat[l * nb_cols_border + c] = (float)(l + c - 2 * border);
}

///////////////////////////////////////////////////////////////////////////
//
// Thread tools
//
///////////////////////////////////////////////////////////////////////////

#ifdef WIN32

void setThreadAffinity(int coreId)
{
    SetThreadAffinityMask(GetCurrentThread(),(1<<coreId)-1);
}

#else

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <stdlib.h>

void setThreadAffinity(int coreId)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
#endif

#ifdef WIN32

#include <atomic>

// Définition d'une barrière
struct barrier {
    unsigned const count;
    std::atomic<unsigned> spaces;
    std::atomic<unsigned> generation;
    barrier(unsigned count_) :
        count(count_), spaces(count_), generation(0)
    {}
    void wait() {
        unsigned const my_generation = generation;
        if (!--spaces) {
            spaces = count;
            ++generation;
        } else {
            while(generation == my_generation);
        }
    }
};

#endif

///////////////////////////////////////////////////////////////////////////
//
// Fonctions outils utilisées pour les tests
//
///////////////////////////////////////////////////////////////////////////

float histoMinVal = -5;
float histoMaxVal = 5;
float seqTime;

static inline void matrixInit(float *mat, int nb_cols, int nb_rows)
{
    int l, c;

    for (l = 0; l < nb_rows; l++)
        for (c = 0; c < nb_cols; c++)
            mat[l * nb_cols + c] = (float)((l + c) % 20 - 10) / 2.f;
}

static inline void vectorInit(float *vec, int size)
{
    int i;

    for (i = 0; i < size; i++) {
        int I = i % 45000;
        int V = I * I;
        vec[i] = (float)(V % 20 - 10) / 2.f;
    }
}

static inline void vectorInit1(float *vec, int size)
{
    int i;

    for (i = 0; i < size; i++) {
        vec[i] = (float)(i % 20 - 10) / 2.f;
    }
}

static inline void vectorInit2(float *vec, int size)
{
    int i;

    for (i = 0; i < size; i++) {
        vec[i] = (float)(i % 22 - 11) / 2.f;
    }
}

float* vectorSeq;

static inline void matVecSeq(float* matrix, float* vectorIn, int size, int loop = 1)
{
    mytimer_t start;

    vectorSeq = new float[size];

    timer_start(&start);

    // Boucle pour obtenir un temps de traitement plus élevé.
    for (int i = 0; i < loop; i++)
    {
        for (int l = 0; l < size; l++)
        {
            float res = 0;
            for (int c = 0; c < size; c++)
            {
                res += matrix[l * size + c] * vectorIn[c];
            }
            vectorSeq[l] = res;
        }
    }

    seqTime = timer_stop(&start);

    printf ("MatVec (seq): %.3fs\n", seqTime);
}

static inline void matVecCheck(float* vectorOut, int size)
{
    printf ("Check results.\n");
    for (int i = 0; i < size; i++)
    {
        if (vectorOut[i] != vectorSeq[i]) {
            printf ("Error at index %d:\n", i);
            printf (" - Computed value: %f\n", vectorOut[i]);
            printf (" - Reference value: %f\n", vectorSeq[i]);
            abort();
        }
    }
    printf ("Results : correct !\n");
}

double dotProdSeq;

static inline void dotProductSeq(float* vectorA, float* vectorB, int size, int loop = 1)
{
    mytimer_t start;

    timer_start(&start);

    // Boucle pour obtenir un temps de traitement plus élevé.
    for (int l = 0; l < loop; l++)
    {
        double res = 0;

        for (int i = 0; i < size; i++)
        {
            res += vectorA[i] * vectorB[i];
        }
        dotProdSeq = res;
    }

    seqTime = timer_stop(&start);

    printf ("DotProduct (seq): %.3fs\n", seqTime);
}

static inline void dotProductCheck(double dotProduct)
{
    printf ("Check results.\n");
    if (dotProduct != dotProdSeq)
    {
        printf ("Error:\n");
        printf (" - Computed value: %f\n", dotProduct);
        printf (" - Reference value: %f\n", dotProdSeq);
        abort();
    }
    printf ("Results : correct !\n");
}

int* histogramSeq;

static inline void computeHistoSeq(float* dataIn, float minVal, float maxVal, int dataSize, int histoSize)
{
    mytimer_t start;
    float step = (maxVal - minVal) / (histoSize - 1);

    histogramSeq = new int[histoSize];

    timer_start(&start);

    for (int i = 0; i < histoSize; i++)
        histogramSeq[i] = 0;

    for (int i = 0; i < dataSize; i++)
    {
        int bin = (int)floor((dataIn[i] - minVal) / step);
        histogramSeq[bin]++;
    }

    seqTime = timer_stop(&start);

    printf ("Histogram (seq): %.3fs\n", seqTime);
}

static inline void histogramCheck(int* histogram, int size)
{
    printf ("Check results.\n");
    for (int i = 0; i < size; i++)
    {
        if (histogram[i] != histogramSeq[i]) {
            printf ("Error at index %d:\n", i);
            printf (" - Computed value: %d\n", histogram[i]);
            printf (" - Reference value: %d\n", histogramSeq[i]);
            abort();
        }
    }
    printf ("Results : correct !\n");
}
