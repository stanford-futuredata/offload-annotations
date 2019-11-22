/**
 * exp.cu
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include <sys/time.h>

// To set dependent parameters.
#include "mkl_vml_functions.h"
#include "mkl.h"

// Extensions that support immediates, etc.
#include <mkl_extensions.h>
#include <vec.h>
#include <generated/generated.h>
#include <composer.h>
#include <omp.h>

typedef struct {
  vec_t data;
  double runtime;
} result_t;

double c05 = 3.0;
double c10 = 1.5;

// Number of iterations.
long outerIters = 1L << 8;
// Piece size for pipelined execution.
long piece_size = 1L << 19;
// Size of the input.
size_t data_size = (1L << 24);

double runtime(struct timeval start) {
  struct timeval end, diff;
  gettimeofday(&end, NULL);
  timersub(&end, &start, &diff);
  return (double)diff.tv_sec + ((double)diff.tv_usec / 1000000.0);
}

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void cudaExp(int n, double *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int innerIters = 1;
    for (int j = 0; j < innerIters; j++) {
      a[i] = exp(a[i]);
    }
  }
}

void run_cuda_nopipe(struct timeval start,
    double* vec, double* h_vec, double* data, double* h_data) {

  int bytes = data_size * sizeof(double);
  checkCuda(cudaMemcpy(vec, h_vec, bytes, cudaMemcpyHostToDevice));
  printf("Transfer H2D: %fs\n", runtime(start));

  int threadsPerBlock = 256;
  int pieces = data_size / piece_size;
  int numBlocks = (piece_size + threadsPerBlock - 1) / threadsPerBlock;

  struct timeval compute_start;
  gettimeofday(&compute_start, NULL);
  for (int i = 0; i < pieces; i++) {
    int o = piece_size * i;
    for (int j = 0; j < outerIters; j++) {
      cudaExp<<<numBlocks, threadsPerBlock>>>(piece_size, &vec[o]);
    }
  }
  cudaDeviceSynchronize();
  printf("Evaluation: %fs\n", runtime(compute_start));

  gettimeofday(&compute_start, NULL);
  checkCuda(cudaMemcpy(h_data, vec, bytes, cudaMemcpyDeviceToHost));
  printf("Transfer D2H: %fs\n", runtime(compute_start));
}

void print_usage(char **argv) {
  fprintf(stderr, "%s -p <log2 piece size> -s <log2 data size> -i <log2 outerIters> -h]\n", argv[0]);
}

void parse_args(int argc, char **argv) {
  int opt;
  while ((opt = getopt(argc, argv, "p:s:i:dh")) != -1) {
    switch (opt) {
      case 'p':
        // piece_size = atol(optarg);
        piece_size = 1L << atol(optarg);
        break;
      case 's':
        data_size = atol(optarg);
        if (data_size > 30 || data_size <= 0) {
          fprintf(stderr, "data size must be 1 <= data_size <= 31\n");
          exit(EXIT_FAILURE);
        }
        data_size = (1L << data_size);
        break;
      case 'i':
        outerIters = 1L << atol(optarg);
        break;
      case 'h':
      default:
        print_usage(argv);
        exit(EXIT_FAILURE);
    }
  }
}

result_t run_cuda_main(vec_t h_vec, vec_t h_data) {
  struct timeval start;
  gettimeofday(&start, NULL);

  // Device allocations for fused
  double *d_vec, *d_data;
  int bytes = data_size * sizeof(double);
  printf("Initializing device...");
  checkCuda(cudaMalloc(&d_vec, bytes));
  checkCuda(cudaMalloc(&d_data, bytes));
  printf("done: %fs\n", runtime(start));

  printf("--------------------\n");
  // Run function
  gettimeofday(&start, NULL);

  result_t result;
  run_cuda_nopipe(start, d_vec, h_vec.data, d_data, h_data.data);
  result.data = h_data;

  // Set runtime
  result.runtime = runtime(start);

  printf("--------------------\n");
  gettimeofday(&start, NULL);

  // Free device allocations
  printf("Freeing device allocations...");
  checkCuda(cudaFree(d_vec));
  checkCuda(cudaFree(d_data));
  printf("done: %fs\n", runtime(start));
  return result;
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // Need to call this before any of the other library functions.
  omp_set_num_threads(1);

  printf("Data Size: %ld Piece Size: %ld\n", data_size, piece_size);

  struct timeval start;
  gettimeofday(&start, NULL);

  // Generate inputs.
  printf("Initializing host...");
  vec_t vec = vvals(data_size, 0.99, false);
  vec_t vdata = vvals(data_size, 0, false);
  printf("done: %fs\n", runtime(start));

  // Determine whether we are using cuda
  result_t result;
  result = run_cuda_main(vec, vdata);

  for (int j = 0; j < 5; j++) {
    long index = rand() % data_size;
    fprintf(stderr, "(%f) ", result.data.data[index]);
  }
  fprintf(stderr, "\n");

  fprintf(stderr, "Checking correctness...\n");
  for (int i = 0; i < data_size; i++) {
    if (result.data.data[0] != result.data.data[i]) {
      printf("Data mismatch at position %d\n", i);
      exit(1);
    }
  }
  fprintf(stderr, "\n");
  fflush(stderr);
  fflush(stdout);
  printf("Runtime: %f s\n", result.runtime);
}

