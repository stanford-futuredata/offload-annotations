/**
 * blackscholes.cu
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

typedef enum {
  UNKNOWN = 0,
  FUSED,
  MKL,
  MKL_COMPOSER,
  MKL_COMPOSER_NOPIPE,
  CUDA_FUSED,
  CUDA_FUSED_NOPIPE,
  CUDA_STREAM,
  CUDA_STREAM_NOPIPE,
} exec_mode_t;

typedef struct {
  vec_t call;
  vec_t put;
  double runtime;
} result_t;

double c05 = 3.0;
double c10 = 1.5;

// Piece size for pipelined execution.
long piece_size = 4096;
// Number of threads.
long threads = 1;
// Log number of cuda streams.
long nstreams = (1L << 4);
// Type of cuda allocation
bool pinned = false;
// Dumps as CSV data if true.
int dump = 0;
// Size of the input.
size_t data_size = (1L << 26);
// Mode to use
exec_mode_t mode;

exec_mode_t get_mode(const char *s) {
  if (strcmp("fused", s) == 0) {
    return FUSED;
  } else if (strcmp("mkl", s) == 0) {
    return MKL;
  } else if (strcmp("mklcomposer", s) == 0) {
    return MKL_COMPOSER;
  } else if (strcmp("cudafused", s) == 0) {
    return CUDA_FUSED;
  } else if (strcmp("cudafused-nopipe", s) == 0) {
    return CUDA_FUSED_NOPIPE;
  } else if (strcmp("cudastream", s) == 0) {
    return CUDA_STREAM;
  } else if (strcmp("cudastream-nopipe", s) == 0) {
    return CUDA_STREAM_NOPIPE;
  } else if (strcmp("nopipe", s) == 0) {
    return MKL_COMPOSER_NOPIPE;
  } else {
    return UNKNOWN;
  }
}

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

result_t run_mkl(vec_t vprice,
    vec_t vstrike,
    vec_t vt,
    vec_t vrate,
    vec_t vvol,
    vec_t vrsig,
    vec_t vvol_sqrt,
    vec_t vtmp,
    vec_t vd1, vec_t vd2, vec_t vcall, vec_t vput) {

  /*
  vec_t vrsig = new_vec(vprice.length, 0);
  vec_t vvol_sqrt = new_vec(vprice.length, 0);
  vec_t vtmp = new_vec(vprice.length, 0);
  vec_t vd1 = new_vec(vprice.length, 0);
  vec_t vd2 = new_vec(vprice.length, 0);

  vec_t vcall = new_vec(vprice.length, 0);
  vec_t vput = new_vec(vprice.length, 0);
  */

  // Assign to pointers since that's what MKL takes.
  double *price = vprice.data;
  double *strike = vstrike.data;
  double *t = vt.data;
  double *rate = vrate.data;
  double *vol = vvol.data;
  double *rsig = vrsig.data;
  double *vol_sqrt = vvol_sqrt.data;
  double *tmp = vtmp.data;
  double *d1 = vd1.data;
  double *d2 = vd2.data;
  double *call = vcall.data;
  double *put = vput.data;
  MKL_INT len = vprice.length;

  // Do the actual MKL computation.
  double invsqrt2 = 1 / sqrt(2);

  // Compute rsig = rate + vol * vol * c05
  vdMul(len, vol, vol, rsig);
  vdMuli(len, rsig, c05, rsig);
  vdAdd(len, rate, rsig, rsig);

  // Compute vol_sqrt = vol * sqrt(t)
  vdSqrt(len, t, vol_sqrt);
  vdMul(len, vol, vol_sqrt, vol_sqrt);

  // Compute d1 = (len, log(price / strike) + rsig * t) / vol_sqrt)
  // This is computed here as:
  //
  // tmp = rsig * t
  // d1 = price / strike
  // d1 = log(len, d1)
  // d1 = add d1 + tmp
  // d1 = d1 / vol_sqrt
  vdMul(len, rsig, t, tmp);         /* Finished with rsig */
  vdDiv(len, price, strike, d1);
  vdLog1p(len, d1, d1);
  vdAdd(len, d1, tmp, d1);          /* Finished with tmp */
  vdDiv(len, d1, vol_sqrt, d1);
  vdSub(len, d1, vol_sqrt, d2);     /* Finished with vol_sqrt */

  // d1 = c05 + c05 * erf(d1 * invsqrt2)
  vdMuli(len, d1, invsqrt2, d1);
  vdErf(len, d1, d1);
  vdMuli(len, d1, c05, d1);
  vdAddi(len, d1, c05, d1);

  // d2 = c05 + c05 * erf(len, d2 * invsqrt2)
  vdMuli(len, d2, invsqrt2, d2);
  vdErf(len, d2, d2);
  vdMuli(len, d2, c05, d2);
  vdAddi(len, d2, c05, d2);

  // Reuse existing buffers
  double *e_rt = vol_sqrt;
  double *tmp2 = rsig;

  // e_rt = exp(len, -rate * t)
  vdMuli(len, rate, -1, e_rt);
  vdMul(len, e_rt, t, e_rt);
  vdExp(len, e_rt, e_rt);

  // call = price * d1 - e_rt * strike * d2
  //
  // tmp = price * d1
  // tmp2 = e_rt * strike * d2
  // call = tmp - tmp2
  vdMul(len, price, d1, tmp);
  vdMul(len, e_rt, strike, tmp2);
  vdMul(len, tmp2, d2, tmp2);
  vdSub(len, tmp, tmp2, call);

  // put = e_rt * strike * (len, c10 - d2) - price * (len, c10 - d1)
  // tmp = e_rt * strike
  // tmp2 = (c10 - d2)
  // put = tmp - tmp2
  // tmp = c10 - d1
  // tmp = price * tmp
  // put = put - tmp
  vdMul(len, e_rt, strike, tmp);
  vdSubvi(len, c10, d2, tmp2);
  vdMul(len, tmp, tmp2, put);
  vdSubvi(len, c10, d1, tmp);
  vdMul(len, price, tmp, tmp);
  vdSub(len, put, tmp, put);

  result_t res;
  res.call = vcall;
  res.put = vput;
  return res;
}

result_t run_mkl_composer_nopipe(vec_t vprice,
    vec_t vstrike,
    vec_t vt,
    vec_t vrate,
    vec_t vvol,
    vec_t vrsig,
    vec_t vvol_sqrt,
    vec_t vtmp,
    vec_t vd1, vec_t vd2, vec_t vcall, vec_t vput) {

  /*
  vec_t vrsig = new_vec(vprice.length, 0);
  vec_t vvol_sqrt = new_vec(vprice.length, 0);
  vec_t vtmp = new_vec(vprice.length, 0);
  vec_t vd1 = new_vec(vprice.length, 0);
  vec_t vd2 = new_vec(vprice.length, 0);

  // Mark these as lazy.
  vec_t vcall = new_vec(vprice.length, 1);
  vec_t vput = new_vec(vprice.length, 1);
  */

  // Assign to pointers since that's what MKL takes.
  double *price = vprice.data;
  double *strike = vstrike.data;
  double *t = vt.data;
  double *rate = vrate.data;
  double *vol = vvol.data;
  double *rsig = vrsig.data;
  double *vol_sqrt = vvol_sqrt.data;
  double *tmp = vtmp.data;
  double *d1 = vd1.data;
  double *d2 = vd2.data;
  double *call = vcall.data;
  double *put = vput.data;
  MKL_INT len = vprice.length;

  // TEMPORARY(&rsig, sizeof(rsig));
  // TEMPORARY(&vol_sqrt, sizeof(vol_sqrt));
  // TEMPORARY(&tmp, sizeof(tmp));
  // TEMPORARY(&d1, sizeof(d1));
  // TEMPORARY(&d2, sizeof(d2));

  // Do the actual MKL computation.
  double invsqrt2 = 1 / sqrt(2);

  // Compute rsig = rate + vol * vol * c05
  c_vdMul(len, vol, vol, rsig);
  composer_execute();
  c_vdMuli(len, rsig, c05, rsig);
  composer_execute();
  c_vdAdd(len, rate, rsig, rsig);
  composer_execute();

  // Compute vol_sqrt = vol * sqrt(t)
  c_vdSqrt(len, t, vol_sqrt);
  composer_execute();
  c_vdMul(len, vol, vol_sqrt, vol_sqrt);
  composer_execute();

  // Compute d1 = (len, log(price / strike) + rsig * t) / vol_sqrt)
  // This is computed here as:
  //
  // tmp = rsig * t
  // d1 = price / strike
  // d1 = log(len, d1)
  // d1 = add d1 + tmp
  // d1 = d1 / vol_sqrt
  c_vdMul(len, rsig, t, tmp);         /* Finished with rsig */
  composer_execute();
  c_vdDiv(len, price, strike, d1);
  composer_execute();
  c_vdLog1p(len, d1, d1);
  composer_execute();
  c_vdAdd(len, d1, tmp, d1);          /* Finished with tmp */
  composer_execute();
  c_vdDiv(len, d1, vol_sqrt, d1);
  composer_execute();
  c_vdSub(len, d1, vol_sqrt, d2);     /* Finished with vol_sqrt */
  composer_execute();

  // d1 = c05 + c05 * erf(d1 * invsqrt2)
  c_vdMuli(len, d1, invsqrt2, d1);
  composer_execute();
  c_vdErf(len, d1, d1);
  composer_execute();
  c_vdMuli(len, d1, c05, d1);
  composer_execute();
  c_vdAddi(len, d1, c05, d1);
  composer_execute();

  // d2 = c05 + c05 * erf(len, d2 * invsqrt2)
  c_vdMuli(len, d2, invsqrt2, d2);
  composer_execute();
  c_vdErf(len, d2, d2);
  composer_execute();
  c_vdMuli(len, d2, c05, d2);
  composer_execute();
  c_vdAddi(len, d2, c05, d2);
  composer_execute();

  // Reuse existing buffers
  double *e_rt = vol_sqrt;
  double *tmp2 = rsig;

  // e_rt = exp(len, -rate * t)
  c_vdMuli(len, rate, -1, e_rt);
  composer_execute();
  c_vdMul(len, e_rt, t, e_rt);
  composer_execute();
  c_vdExp(len, e_rt, e_rt);
  composer_execute();

  // call = price * d1 - e_rt * strike * d2
  //
  // tmp = price * d1
  // tmp2 = e_rt * strike * d2
  // call = tmp - tmp2
  c_vdMul(len, price, d1, tmp);
  composer_execute();
  c_vdMul(len, e_rt, strike, tmp2);
  composer_execute();
  c_vdMul(len, tmp2, d2, tmp2);
  composer_execute();
  c_vdSub(len, tmp, tmp2, call);

  // put = e_rt * strike * (len, c10 - d2) - price * (len, c10 - d1)
  // tmp = e_rt * strike
  // tmp2 = (c10 - d2)
  // put = tmp - tmp2
  // tmp = c10 - d1
  // tmp = price * tmp
  // put = put - tmp
  c_vdMul(len, e_rt, strike, tmp);
  composer_execute();
  c_vdSubvi(len, c10, d2, tmp2);
  composer_execute();
  c_vdMul(len, tmp, tmp2, put);
  composer_execute();
  c_vdSubvi(len, c10, d1, tmp);
  composer_execute();
  c_vdMul(len, price, tmp, tmp);
  composer_execute();
  c_vdSub(len, put, tmp, put);
  composer_execute();

  result_t res;
  res.call = vcall;
  res.put = vput;
  return res;
}

result_t run_mkl_composer(vec_t vprice,
    vec_t vstrike,
    vec_t vt,
    vec_t vrate,
    vec_t vvol,
    vec_t vrsig,
    vec_t vvol_sqrt,
    vec_t vtmp,
    vec_t vd1, vec_t vd2, vec_t vcall, vec_t vput) {

  /*
  vec_t vrsig = new_vec(vprice.length, 0);
  vec_t vvol_sqrt = new_vec(vprice.length, 0);
  vec_t vtmp = new_vec(vprice.length, 0);
  vec_t vd1 = new_vec(vprice.length, 0);
  vec_t vd2 = new_vec(vprice.length, 0);

  // Mark these as lazy.
  vec_t vcall = new_vec(vprice.length, 1);
  vec_t vput = new_vec(vprice.length, 1);
  */

  // Assign to pointers since that's what MKL takes.
  double *price = vprice.data;
  double *strike = vstrike.data;
  double *t = vt.data;
  double *rate = vrate.data;
  double *vol = vvol.data;
  double *rsig = vrsig.data;
  double *vol_sqrt = vvol_sqrt.data;
  double *tmp = vtmp.data;
  double *d1 = vd1.data;
  double *d2 = vd2.data;
  double *call = vcall.data;
  double *put = vput.data;
  MKL_INT len = vprice.length;

  // TEMPORARY(&rsig, sizeof(rsig));
  // TEMPORARY(&vol_sqrt, sizeof(vol_sqrt));
  // TEMPORARY(&tmp, sizeof(tmp));
  // TEMPORARY(&d1, sizeof(d1));
  // TEMPORARY(&d2, sizeof(d2));

  // Do the actual MKL computation.
  double invsqrt2 = 1 / sqrt(2);

  // Compute rsig = rate + vol * vol * c05
  c_vdMul(len, vol, vol, rsig);
  c_vdMuli(len, rsig, c05, rsig);
  c_vdAdd(len, rate, rsig, rsig);

  // Compute vol_sqrt = vol * sqrt(t)
  c_vdSqrt(len, t, vol_sqrt);
  c_vdMul(len, vol, vol_sqrt, vol_sqrt);

  // Compute d1 = (len, log(price / strike) + rsig * t) / vol_sqrt)
  // This is computed here as:
  //
  // tmp = rsig * t
  // d1 = price / strike
  // d1 = log(len, d1)
  // d1 = add d1 + tmp
  // d1 = d1 / vol_sqrt
  c_vdMul(len, rsig, t, tmp);         /* Finished with rsig */
  c_vdDiv(len, price, strike, d1);
  c_vdLog1p(len, d1, d1);
  c_vdAdd(len, d1, tmp, d1);          /* Finished with tmp */
  c_vdDiv(len, d1, vol_sqrt, d1);
  c_vdSub(len, d1, vol_sqrt, d2);     /* Finished with vol_sqrt */

  // d1 = c05 + c05 * erf(d1 * invsqrt2)
  c_vdMuli(len, d1, invsqrt2, d1);
  c_vdErf(len, d1, d1);
  c_vdMuli(len, d1, c05, d1);
  c_vdAddi(len, d1, c05, d1);

  // d2 = c05 + c05 * erf(len, d2 * invsqrt2)
  c_vdMuli(len, d2, invsqrt2, d2);
  c_vdErf(len, d2, d2);
  c_vdMuli(len, d2, c05, d2);
  c_vdAddi(len, d2, c05, d2);

  // Reuse existing buffers
  double *e_rt = vol_sqrt;
  double *tmp2 = rsig;

  // e_rt = exp(len, -rate * t)
  c_vdMuli(len, rate, -1, e_rt);
  c_vdMul(len, e_rt, t, e_rt);
  c_vdExp(len, e_rt, e_rt);

  // call = price * d1 - e_rt * strike * d2
  //
  // tmp = price * d1
  // tmp2 = e_rt * strike * d2
  // call = tmp - tmp2
  c_vdMul(len, price, d1, tmp);
  c_vdMul(len, e_rt, strike, tmp2);
  c_vdMul(len, tmp2, d2, tmp2);
  c_vdSub(len, tmp, tmp2, call);

  // put = e_rt * strike * (len, c10 - d2) - price * (len, c10 - d1)
  // tmp = e_rt * strike
  // tmp2 = (c10 - d2)
  // put = tmp - tmp2
  // tmp = c10 - d1
  // tmp = price * tmp
  // put = put - tmp
  c_vdMul(len, e_rt, strike, tmp);
  c_vdSubvi(len, c10, d2, tmp2);
  c_vdMul(len, tmp, tmp2, put);
  c_vdSubvi(len, c10, d1, tmp);
  c_vdMul(len, price, tmp, tmp);
  c_vdSub(len, put, tmp, put);

  result_t res;
  res.call = vcall;
  res.put = vput;
  return res;
}

result_t run_fused(vec_t price,
    vec_t strike,
    vec_t t,
    vec_t rate,
    vec_t vol) {

  vec_t call = new_vec_nolazy(price.length);
  vec_t put = new_vec_nolazy(price.length);

  const double invsqrt2 = 1 / sqrt(2);

  for (size_t i = 0; i < price.length; i++) {
    double rsig = rate.data[i] + (vol.data[i] * vol.data[i]) * c05;
    double vol_sqrt = vol.data[i] * sqrt(t.data[i]);
    double d1 = (log2(price.data[i] / strike.data[i]) + rsig * t.data[i]) / vol_sqrt;
    double d2 = d1 - vol_sqrt;
    d1 = c05 + c05 * erf(d1 * invsqrt2);
    d2 = c05 + c05 * erf(d2 * invsqrt2);

    double e_rt = exp(-rate.data[i] * t.data[i]);
    call.data[i] = price.data[i] * d1 - e_rt * strike.data[i] * d2;
    put.data[i] = e_rt * strike.data[i] * (c10 - d2) - price.data[i] * (c10 - d1);
  }

  result_t res;
  res.call = call;
  res.put = put;
  return res;
}

__global__ void cudaAdd(int n, double *a, double *b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = a[i] + b[i]; }
}
__global__ void cudaDiv(int n, double *a, double *b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = a[i] / b[i]; }
}
__global__ void cudaMul(int n, double *a, double *b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = a[i] * b[i]; }
}
__global__ void cudaSub(int n, double *a, double *b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = a[i] - b[i]; }
}
__global__ void cudaErf(int n, double *a, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = erf(out[i]); }
}
__global__ void cudaExp(int n, double *a, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = exp(a[i]); }
}
__global__ void cudaLog1p(int n, double *a, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = log2(a[i]); }
}
__global__ void cudaSqrt(int n, double *a, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = sqrt(a[i]); }
}
__global__ void cudaAddi(int n, double *a, double b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = a[i] + b; }
}
__global__ void cudaMuli(int n, double *a, double b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = a[i] * b; }
}
__global__ void cudaSubvi(int n, double a, double *b, double *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { out[i] = a - b[i]; }
}

void run_cuda_nopipe(struct timeval start,
    double* price,
    double* strike,
    double* t,
    double* rate,
    double* vol,
    double* rsig,
    double* vol_sqrt,
    double* tmp,
    double* d1, double* d2, double* call, double* put,
    double* h_price,
    double* h_strike,
    double* h_t,
    double* h_rate,
    double* h_vol,
    double* h_call, double* h_put) {

  fprintf(stderr, "Copying data from host to device...");
  fflush(stdout);
  int bytes = data_size * sizeof(double);
  checkCuda(cudaMemcpy(price, h_price, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(strike, h_strike, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(t, h_t, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(rate, h_rate, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(vol, h_vol, bytes, cudaMemcpyHostToDevice));
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  // int n = data_size;
  int n = piece_size;  // less than 1<<27
  int pieces = data_size / n;
  double invsqrt2 = 1 / sqrt(2);

  struct timeval compute_start;
  gettimeofday(&compute_start, NULL);
  for (int i = 0; i < pieces; i++) {
    int o = n * i;
    // Compute rsig = rate + vol * vol * c05
    cudaMul<<<(n+255)/256, 256>>>(n, &vol[o], &vol[o], &rsig[o]);
    cudaMuli<<<(n+255)/256, 256>>>(n, &rsig[o], c05, &rsig[o]);
    cudaAdd<<<(n+255)/256, 256>>>(n, &rate[o], &rsig[o], &rsig[o]);

    // Compute vol_sqrt = vol * sqrt(t)
    cudaSqrt<<<(n+255)/256, 256>>>(n, &t[o], &vol_sqrt[o]);
    cudaMul<<<(n+255)/256, 256>>>(n, &vol[o], &vol_sqrt[o], &vol_sqrt[o]);

    // Compute d1 = (len, log(price / strike) + rsig * t) / vol_sqrt)
    // This is computed here as:
    //
    // tmp = rsig * t
    // d1 = price / strike
    // d1 = log(len, d1)
    // d1 = add d1 + tmp
    // d1 = d1 / vol_sqrt
    cudaMul<<<(n+255)/256, 256>>>(n, &rsig[o], &t[o], &tmp[o]);         /* Finished with rsig */
    cudaDiv<<<(n+255)/256, 256>>>(n, &price[o], &strike[o], &d1[o]);
    cudaLog1p<<<(n+255)/256, 256>>>(n, &d1[o], &d1[o]);
    cudaAdd<<<(n+255)/256, 256>>>(n, &d1[o], &tmp[o], &d1[o]);          /* Finished with tmp */
    cudaDiv<<<(n+255)/256, 256>>>(n, &d1[o], &vol_sqrt[o], &d1[o]);
    cudaSub<<<(n+255)/256, 256>>>(n, &d1[o], &vol_sqrt[o], &d2[o]);     /* Finished with vol_sqrt */

    // d1 = c05 + c05 * erf(d1 * invsqrt2)
    cudaMuli<<<(n+255)/256, 256>>>(n, &d1[o], invsqrt2, &d1[o]);
    cudaErf<<<(n+255)/256, 256>>>(n, &d1[o], &d1[o]);
    cudaMuli<<<(n+255)/256, 256>>>(n, &d1[o], c05, &d1[o]);
    cudaAddi<<<(n+255)/256, 256>>>(n, &d1[o], c05, &d1[o]);

    // d2 = c05 + c05 * erf(len, d2 * invsqrt2)
    cudaMuli<<<(n+255)/256, 256>>>(n, &d2[o], invsqrt2, &d2[o]);
    cudaErf<<<(n+255)/256, 256>>>(n, &d2[o], &d2[o]);
    cudaMuli<<<(n+255)/256, 256>>>(n, &d2[o], c05, &d2[o]);
    cudaAddi<<<(n+255)/256, 256>>>(n, &d2[o], c05, &d2[o]);

    // Reuse existing buffers
    double *e_rt = vol_sqrt;
    double *tmp2 = rsig;

    // e_rt = exp(len, -rate * t)
    cudaMuli<<<(n+255)/256, 256>>>(n, &rate[o], -1, &e_rt[o]);
    cudaMul<<<(n+255)/256, 256>>>(n, &e_rt[o], &t[o], &e_rt[o]);
    cudaExp<<<(n+255)/256, 256>>>(n, &e_rt[o], &e_rt[o]);

    // call = price * d1 - e_rt * strike * d2
    //
    // tmp = price * d1
    // tmp2 = e_rt * strike * d2
    // call = tmp - tmp2
    cudaMul<<<(n+255)/256, 256>>>(n, &price[o], &d1[o], &tmp[o]);
    cudaMul<<<(n+255)/256, 256>>>(n, &e_rt[o], &strike[o], &tmp2[o]);
    cudaMul<<<(n+255)/256, 256>>>(n, &tmp2[o], &d2[o], &tmp2[o]);
    cudaSub<<<(n+255)/256, 256>>>(n, &tmp[o], &tmp2[o], &call[o]);

    // put = e_rt * strike * (len, c10 - d2) - price * (len, c10 - d1)
    // tmp = e_rt * strike
    // tmp2 = (c10 - d2)
    // put = tmp - tmp2
    // tmp = c10 - d1
    // tmp = price * tmp
    // put = put - tmp
    cudaMul<<<(n+255)/256, 256>>>(n, &e_rt[o], &strike[o], &tmp[o]);
    cudaSubvi<<<(n+255)/256, 256>>>(n, c10, &d2[o], &tmp2[o]);
    cudaMul<<<(n+255)/256, 256>>>(n, &tmp[o], &tmp2[o], &put[o]);
    cudaSubvi<<<(n+255)/256, 256>>>(n, c10, &d1[o], &tmp[o]);
    cudaMul<<<(n+255)/256, 256>>>(n, &price[o], &tmp[o], &tmp[o]);
    cudaSub<<<(n+255)/256, 256>>>(n, &put[o], &tmp[o], &put[o]);
  }
  fprintf(stderr, "Compute time: %fs\n", runtime(compute_start));
  fflush(stdout);

  fprintf(stderr, "Copying data from device to host...");
  fflush(stdout);
  checkCuda(cudaMemcpy(h_call, call, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_put, put, bytes, cudaMemcpyDeviceToHost));

  // Required to force Composer's lazy evaluation for timing (not true for CUDA).
  double first_call = h_call[0];
  double first_put = h_put[0];
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);
  fprintf(stderr, "First call value: %f\n", first_call);
  fprintf(stderr, "First put value: %f\n", first_put);
  fflush(stderr);
}

__global__
void op_parallel(int n,
    double invsqrt2,
    double *price,
    double *strike,
    double *t,
    double *rate,
    double *vol,
    double *call,
    double *put) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double rsig = rate[i] + (vol[i] * vol[i]) * 3.0;
    double vol_sqrt = vol[i] * sqrt(t[i]);
    double d1 = (log2(price[i] / strike[i]) + rsig * t[i]) / vol_sqrt;
    double d2 = d1 - vol_sqrt;
    d1 = 3.0 + 3.0 * erf(d1 * invsqrt2);
    d2 = 3.0 + 3.0 * erf(d2 * invsqrt2);

    double e_rt = exp(-rate[i] * t[i]);
    call[i] = price[i] * d1 - e_rt * strike[i] * d2;
    put[i] = e_rt * strike[i] * (1.5 - d2) - price[i] * (1.5 - d1);
  }
}

__global__
void op_parallel_piece(int n,
    int pieceSize,
    double invsqrt2,
    double *price,
    double *strike,
    double *t,
    double *rate,
    double *vol,
    double *call,
    double *put) {
  int pieceStart = (blockIdx.x * blockDim.x + threadIdx.x) * pieceSize;
  int pieceEnd = min(pieceStart + pieceSize, n);
  for (int i = pieceStart; i < pieceEnd; i++) {
    double rsig = rate[i] + (vol[i] * vol[i]) * 3.0;
    double vol_sqrt = vol[i] * sqrt(t[i]);
    double d1 = (log2(price[i] / strike[i]) + rsig * t[i]) / vol_sqrt;
    double d2 = d1 - vol_sqrt;
    d1 = 3.0 + 3.0 * erf(d1 * invsqrt2);
    d2 = 3.0 + 3.0 * erf(d2 * invsqrt2);

    double e_rt = exp(-rate[i] * t[i]);
    call[i] = price[i] * d1 - e_rt * strike[i] * d2;
    put[i] = e_rt * strike[i] * (1.5 - d2) - price[i] * (1.5 - d1);
  }
}

void run_cuda_fused(
    struct timeval start,
    double* d_price,
    double* d_strike,
    double* d_t,
    double* d_rate,
    double* d_vol,
    double* d_call,
    double* d_put,
    double* h_price,
    double* h_strike,
    double* h_t,
    double* h_rate,
    double* h_vol,
    double* h_call,
    double* h_put) {

  fprintf(stderr, "Copying data from host to device...");
  fflush(stdout);
  int bytes = data_size * sizeof(double);
  checkCuda(cudaMemcpy(d_price, h_price, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_strike, h_strike, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_t, h_t, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_rate, h_rate, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_vol, h_vol, bytes, cudaMemcpyHostToDevice));
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  // do the computation
  const double invsqrt2 = 1 / sqrt(2);
  const int elementsPerBlock = (data_size+255)/256;
  const int threadsPerBlock = (elementsPerBlock + piece_size - 1) / piece_size;
  if (piece_size == 1) {
    op_parallel<<<(data_size+255)/256, 256>>>(data_size,
      invsqrt2,
      d_price,
      d_strike,
      d_t,
      d_rate,
      d_vol,
      d_call,
      d_put);
  } else {
    op_parallel_piece<<<threadsPerBlock, 256>>>(data_size,
      piece_size,
      invsqrt2,
      d_price,
      d_strike,
      d_t,
      d_rate,
      d_vol,
      d_call,
      d_put);
  }

  fprintf(stderr, "Copying data from device to host...");
  fflush(stdout);
  checkCuda(cudaMemcpy(h_call, d_call, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_put, d_put, bytes, cudaMemcpyDeviceToHost));

  // Required to force Composer's lazy evaluation for timing (not true for CUDA).
  double first_call = h_call[0];
  double first_put = h_put[0];
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);
  fprintf(stderr, "First call value: %f\n", first_call);
  fprintf(stderr, "First put value: %f\n", first_put);
  fflush(stderr);
}

void run_cuda_stream(
    struct timeval start,
    double* d_price,
    double* d_strike,
    double* d_t,
    double* d_rate,
    double* d_vol,
    double* d_call,
    double* d_put,
    double* h_price,
    double* h_strike,
    double* h_t,
    double* h_rate,
    double* h_vol,
    double* h_call,
    double* h_put) {

  fprintf(stderr, "Creating cuda streams...");
  fflush(stdout);
  const int blockSize = (1L << 8);
  const int nTotalStreams = 3;
  assert(data_size % nstreams == 0);

  int streamSize = data_size / nstreams;
  int streamBytes = streamSize * sizeof(double);
  cudaStream_t *streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
  for (int i = 0; i < nTotalStreams; i++) {
    checkCuda(cudaStreamCreate(&streams[i]));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  fprintf(stderr, "Copy data from host to device...");
  fflush(stdout);
  const double invsqrt2 = 1 / sqrt(2);
  const int threadsPerBlock = (streamSize / blockSize + piece_size - 1) / piece_size;
  for (int i = 0; i < nstreams; i++) {
    int o = i * streamSize;
    cudaStream_t stream = streams[i % nTotalStreams];
    checkCuda(cudaMemcpyAsync(&d_price[o], &h_price[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&d_strike[o], &h_strike[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&d_t[o], &h_t[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&d_rate[o], &h_rate[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&d_vol[o], &h_vol[o], streamBytes, cudaMemcpyHostToDevice, stream));
  // }
  // fprintf(stderr, "done: %fs\n", runtime(start));

  // fprintf(stderr, "Launching async cuda calls...");
  // fflush(stdout);
  // for (int i = 0; i < nstreams; i++) {
  //   int o = i * streamSize;

    if (piece_size == 1) {
      op_parallel<<<streamSize/blockSize, blockSize, 0, stream>>>(streamSize,
        invsqrt2,
        &d_price[o],
        &d_strike[o],
        &d_t[o],
        &d_rate[o],
        &d_vol[o],
        &d_call[o],
        &d_put[o]);
    } else {
      op_parallel_piece<<<threadsPerBlock, blockSize, 0, stream>>>(streamSize,
        piece_size,
        invsqrt2,
        &d_price[o],
        &d_strike[o],
        &d_t[o],
        &d_rate[o],
        &d_vol[o],
        &d_call[o],
        &d_put[o]);
    }

    checkCuda(cudaMemcpyAsync(&h_call[o], &d_call[o], streamBytes, cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaMemcpyAsync(&h_put[o], &d_put[o], streamBytes, cudaMemcpyDeviceToHost, stream));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));

  // Fully execute all async calls.
  fprintf(stderr, "Executing async cuda calls...");
  fflush(stdout);
  cudaDeviceSynchronize();
  double first_call = h_call[0];
  double first_put = h_put[0];
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);
  fprintf(stderr, "First call value: %f\n", first_call);
  fprintf(stderr, "First put value: %f\n", first_put);
  fflush(stderr);

  fprintf(stderr, "Destroying cuda streams...");
  fflush(stdout);
  for (int i = 0; i < nTotalStreams; i++) {
    checkCuda(cudaStreamDestroy(streams[i]));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);
}

void run_cuda_stream_nopipe(struct timeval start,
    double* price,
    double* strike,
    double* t,
    double* rate,
    double* vol,
    double* rsig,
    double* vol_sqrt,
    double* tmp,
    double* d1, double* d2, double* call, double* put,
    double* h_price,
    double* h_strike,
    double* h_t,
    double* h_rate,
    double* h_vol,
    double* h_call, double* h_put) {

  fprintf(stderr, "Creating cuda streams...");
  fflush(stdout);
  const int blockSize = (1L << 8);
  const int nTotalStreams = 3;
  assert(data_size % nstreams == 0);

  int streamSize = data_size / nstreams;
  int streamBytes = streamSize * sizeof(double);
  int numBlocks = streamSize / blockSize;
  cudaStream_t *streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
  for (int i = 0; i < nTotalStreams; i++) {
    checkCuda(cudaStreamCreate(&streams[i]));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  double invsqrt2 = 1 / sqrt(2);
  for (int i = 0; i < nstreams; i++) {
    int o = i * streamSize;
    cudaStream_t stream = streams[i % nTotalStreams];
    checkCuda(cudaMemcpyAsync(&price[o], &h_price[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&strike[o], &h_strike[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&t[o], &h_t[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&rate[o], &h_rate[o], streamBytes, cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(&vol[o], &h_vol[o], streamBytes, cudaMemcpyHostToDevice, stream));

    // Compute rsig = rate + vol * vol * c05
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &vol[o], &vol[o], &rsig[o]);
    cudaMuli<<<numBlocks, blockSize, 0, stream>>>(streamSize, &rsig[o], c05, &rsig[o]);
    cudaAdd<<<numBlocks, blockSize, 0, stream>>>(streamSize, &rate[o], &rsig[o], &rsig[o]);

    // Compute vol_sqrt = vol * sqrt(t)
    cudaSqrt<<<numBlocks, blockSize, 0, stream>>>(streamSize, &t[o], &vol_sqrt[o]);
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &vol[o], &vol_sqrt[o], &vol_sqrt[o]);

    // Compute d1 = (len, log(price / strike) + rsig * t) / vol_sqrt)
    // This is computed here as:
    //
    // tmp = rsig * t
    // d1 = price / strike
    // d1 = log(len, d1)
    // d1 = add d1 + tmp
    // d1 = d1 / vol_sqrt
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &rsig[o], &t[o], &tmp[o]);         /* Finished with rsig */
    cudaDiv<<<numBlocks, blockSize, 0, stream>>>(streamSize, &price[o], &strike[o], &d1[o]);
    cudaLog1p<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], &d1[o]);
    cudaAdd<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], &tmp[o], &d1[o]);          /* Finished with tmp */
    cudaDiv<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], &vol_sqrt[o], &d1[o]);
    cudaSub<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], &vol_sqrt[o], &d2[o]);     /* Finished with vol_sqrt */

    // d1 = c05 + c05 * erf(d1 * invsqrt2)
    cudaMuli<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], invsqrt2, &d1[o]);
    cudaErf<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], &d1[o]);
    cudaMuli<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], c05, &d1[o]);
    cudaAddi<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d1[o], c05, &d1[o]);

    // d2 = c05 + c05 * erf(len, d2 * invsqrt2)
    cudaMuli<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d2[o], invsqrt2, &d2[o]);
    cudaErf<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d2[o], &d2[o]);
    cudaMuli<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d2[o], c05, &d2[o]);
    cudaAddi<<<numBlocks, blockSize, 0, stream>>>(streamSize, &d2[o], c05, &d2[o]);

    // Reuse existing buffers
    double *e_rt = vol_sqrt;
    double *tmp2 = rsig;

    // e_rt = exp(len, -rate * t)
    cudaMuli<<<numBlocks, blockSize, 0, stream>>>(streamSize, &rate[o], -1, &e_rt[o]);
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &e_rt[o], &t[o], &e_rt[o]);
    cudaExp<<<numBlocks, blockSize, 0, stream>>>(streamSize, &e_rt[o], &e_rt[o]);

    // call = price * d1 - e_rt * strike * d2
    //
    // tmp = price * d1
    // tmp2 = e_rt * strike * d2
    // call = tmp - tmp2
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &price[o], &d1[o], &tmp[o]);
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &e_rt[o], &strike[o], &tmp2[o]);
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &tmp2[o], &d2[o], &tmp2[o]);
    cudaSub<<<numBlocks, blockSize, 0, stream>>>(streamSize, &tmp[o], &tmp2[o], &call[o]);

    // put = e_rt * strike * (len, c10 - d2) - price * (len, c10 - d1)
    // tmp = e_rt * strike
    // tmp2 = (c10 - d2)
    // put = tmp - tmp2
    // tmp = c10 - d1
    // tmp = price * tmp
    // put = put - tmp
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &e_rt[o], &strike[o], &tmp[o]);
    cudaSubvi<<<numBlocks, blockSize, 0, stream>>>(streamSize, c10, &d2[o], &tmp2[o]);
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &tmp[o], &tmp2[o], &put[o]);
    cudaSubvi<<<numBlocks, blockSize, 0, stream>>>(streamSize, c10, &d1[o], &tmp[o]);
    cudaMul<<<numBlocks, blockSize, 0, stream>>>(streamSize, &price[o], &tmp[o], &tmp[o]);
    cudaSub<<<numBlocks, blockSize, 0, stream>>>(streamSize, &put[o], &tmp[o], &put[o]);

    checkCuda(cudaMemcpyAsync(&h_call[o], &call[o], streamBytes, cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaMemcpyAsync(&h_put[o], &put[o], streamBytes, cudaMemcpyDeviceToHost, stream));
  }

  // Fully execute all async calls.
  fprintf(stderr, "Executing async cuda calls...");
  fflush(stdout);
  cudaDeviceSynchronize();
  double first_call = h_call[0];
  double first_put = h_put[0];
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  fprintf(stderr, "First call value: %f\n", first_call);
  fprintf(stderr, "First put value: %f\n", first_put);
  fflush(stderr);

  fprintf(stderr, "Destroying cuda streams...");
  fflush(stdout);
  for (int i = 0; i < nTotalStreams; i++) {
    checkCuda(cudaStreamDestroy(streams[i]));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);
}

int power_of_two(long x) {
  return x && !(x & (x - 1));
}

void print_usage(char **argv) {
  fprintf(stderr, "%s -m <mode> [-t <threads> -p <piece size> -s <log2 data size> "
                  "-n <cuda streams> -a <allocation_type> -h]\n", argv[0]);
  fprintf(stderr, "Available modes:\n");
  fprintf(stderr, "\tfused\n"
                  "\tmkl\n"
                  "\tmklcomposer\n"
                  "\tnopipe\n"
                  "\tcudafused\n"
                  "\tcudafused-nopipe\n"
                  "\tcudastream\n"
                  "\tcudastream-nopipe\n"
                  );
  fprintf(stderr, "Available allocation types:\n");
  fprintf(stderr, "\tnaive\n"
                  "\tpinned\n"
                  );
}

void parse_args(int argc, char **argv) {
  int opt;
  while ((opt = getopt(argc, argv, "m:t:p:s:n:a:dh")) != -1) {
    switch (opt) {
      case 'm':
        mode = get_mode(optarg);
        if (mode == UNKNOWN) {
          print_usage(argv);
          exit(EXIT_FAILURE);
        }
        break;
      case 'd':
        dump = 1;
        break;
      case 'p':
        // piece_size = atol(optarg);
        piece_size = 1L << atol(optarg);
        break;
      case 't':
        threads = atol(optarg);
        break;
      case 'n':
        nstreams = 1L << atol(optarg);
        break;
      case 's':
        data_size = atol(optarg);
        if (data_size > 30 || data_size <= 0) {
          fprintf(stderr, "data size must be 1 <= data_size <= 31\n");
          exit(EXIT_FAILURE);
        }
        data_size = (1L << data_size);
        break;
      case 'a':
        if (strcmp("naive", optarg) == 0) {
          pinned = false;
        } else if (strcmp("pinned", optarg) == 0) {
          pinned = true;
        } else {
          print_usage(argv);
          exit(EXIT_FAILURE);
        }
        break;
      case 'h':
      default:
        print_usage(argv);
        exit(EXIT_FAILURE);
    }
  }
}

result_t run_mkl_main(vec_t price,
    vec_t strike,
    vec_t t,
    vec_t rate,
    vec_t vol,
    vec_t vrsig,
    vec_t vvol_sqrt,
    vec_t vtmp,
    vec_t vd1,
    vec_t vd2,
    vec_t vcall,
    vec_t vput) {
  // Run function
  struct timeval start;
  gettimeofday(&start, NULL);
  result_t result;
  switch (mode) {
    case FUSED:
      result = run_fused(price, strike, t, rate, vol);
      break;
    case MKL:
      result = run_mkl(price, strike, t, rate, vol, vrsig, vvol_sqrt, vtmp, vd1, vd2, vcall, vput);
      break;
    case MKL_COMPOSER:
      result = run_mkl_composer(price, strike, t, rate, vol, vrsig, vvol_sqrt, vtmp, vd1, vd2, vcall, vput);
      break;
    case MKL_COMPOSER_NOPIPE:
      result = run_mkl_composer_nopipe(price, strike, t, rate, vol, vrsig, vvol_sqrt, vtmp, vd1, vd2, vcall, vput);
      break;

    case UNKNOWN:
    default:
      fprintf(stderr, "unsupported case");
      exit(EXIT_FAILURE);
  }

  fprintf(stderr, "Evaluating lazy calls...\n");
  fflush(stderr);

  // Required to force Composer's lazy evaluation for timing.
  double first_call = result.call.data[0];
  double first_put = result.put.data[0];
  fprintf(stderr, "First call value: %f\n", first_call);
  fprintf(stderr, "First put value: %f\n", first_put);
  fflush(stderr);

  result.runtime = runtime(start);
  return result;
}

result_t run_cuda_main(vec_t h_pricePageable,
    vec_t h_strikePageable,
    vec_t h_tPageable,
    vec_t h_ratePageable,
    vec_t h_volPageable,
    vec_t h_callPageable,
    vec_t h_putPageable) {
  struct timeval start;
  gettimeofday(&start, NULL);

  // Device allocations for fused
  double *d_price, *d_strike, *d_t, *d_rate, *d_vol, *d_call, *d_put;
  double *h_price, *h_strike, *h_t, *h_rate, *h_vol, *h_call, *h_put;
  double *d_rsig, *d_vol_sqrt, *d_tmp, *d_d1, *d_d2;
  int bytes = data_size * sizeof(double);
  fprintf(stderr, "Allocating device memory...");
  fflush(stdout);
  checkCuda(cudaMalloc(&d_price, bytes));
  checkCuda(cudaMalloc(&d_strike, bytes));
  checkCuda(cudaMalloc(&d_t, bytes));
  checkCuda(cudaMalloc(&d_rate, bytes));
  checkCuda(cudaMalloc(&d_vol, bytes));
  checkCuda(cudaMalloc(&d_call, bytes));
  checkCuda(cudaMalloc(&d_put, bytes));

  if (mode != CUDA_FUSED) {
    checkCuda(cudaMalloc(&d_rsig, bytes));
    checkCuda(cudaMalloc(&d_vol_sqrt, bytes));
    checkCuda(cudaMalloc(&d_tmp, bytes));
    checkCuda(cudaMalloc(&d_d1, bytes));
    checkCuda(cudaMalloc(&d_d2, bytes));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  // Re-allocate host memory if pinned
  if (pinned) {
    fprintf(stderr, "Reallocating pinned host memory...");
    fflush(stdout);
    checkCuda(cudaMallocHost(&h_price, bytes));
    checkCuda(cudaMallocHost(&h_strike, bytes));
    checkCuda(cudaMallocHost(&h_t, bytes));
    checkCuda(cudaMallocHost(&h_rate, bytes));
    checkCuda(cudaMallocHost(&h_vol, bytes));
    checkCuda(cudaMallocHost(&h_call, bytes));
    checkCuda(cudaMallocHost(&h_put, bytes));
    fprintf(stderr, "done: %fs\n", runtime(start));
  } else {
    h_price = h_pricePageable.data;
    h_strike = h_strikePageable.data;
    h_t = h_tPageable.data;
    h_rate = h_ratePageable.data;
    h_vol = h_volPageable.data;
    h_call = h_callPageable.data;
    h_put = h_putPageable.data;
  }

  fprintf(stderr, "--------------------\n");
  // Run function
  gettimeofday(&start, NULL);

  // Copy data from pageable to pinned
  if (pinned) {
    fprintf(stderr, "Copying pageable host memory to pinned host memory...");
    fflush(stdout);
    memcpy(h_price, h_pricePageable.data, bytes);
    memcpy(h_strike, h_strikePageable.data, bytes);
    memcpy(h_t, h_tPageable.data, bytes);
    memcpy(h_rate, h_ratePageable.data, bytes);
    memcpy(h_vol, h_volPageable.data, bytes);
    fprintf(stderr, "done: %fs\n", runtime(start));
    fflush(stdout);
  }

  result_t result;
  switch (mode) {
    case CUDA_FUSED:
      run_cuda_fused(start,
                     d_price, d_strike, d_t, d_rate, d_vol, d_call, d_put,
                     h_price, h_strike, h_t, h_rate, h_vol, h_call, h_put);
      break;
    case CUDA_FUSED_NOPIPE:
      run_cuda_nopipe(start,
                      d_price, d_strike, d_t, d_rate, d_vol,
                      d_rsig, d_vol_sqrt, d_tmp, d_d1, d_d2, d_call, d_put,
                      h_price, h_strike, h_t, h_rate, h_vol, h_call, h_put);
      break;
    case CUDA_STREAM:
      run_cuda_stream(start,
                      d_price, d_strike, d_t, d_rate, d_vol, d_call, d_put,
                      h_price, h_strike, h_t, h_rate, h_vol, h_call, h_put);
      break;
    case CUDA_STREAM_NOPIPE:
      run_cuda_stream_nopipe(start,
                      d_price, d_strike, d_t, d_rate, d_vol,
                      d_rsig, d_vol_sqrt, d_tmp, d_d1, d_d2, d_call, d_put,
                      h_price, h_strike, h_t, h_rate, h_vol, h_call, h_put);
      break;

    case UNKNOWN:
    default:
      fprintf(stderr, "unsupported case");
      exit(EXIT_FAILURE);
  }

  if (pinned) {
    fprintf(stderr, "Copying pinned host memory to pageable host memory...");
    fflush(stdout);
    memcpy(h_callPageable.data, h_call, bytes);
    memcpy(h_putPageable.data, h_put, bytes);
    fprintf(stderr, "done: %fs\n", runtime(start));
    fflush(stdout);
  }
  result.call = h_callPageable;
  result.put = h_putPageable;

  // Set runtime
  result.runtime = runtime(start);

  fprintf(stderr, "--------------------\n");
  gettimeofday(&start, NULL);

  // Free pinned memory
  fprintf(stderr, "Freeing pinned memory...");
  fflush(stdout);
  if (pinned) {
    checkCuda(cudaFreeHost(h_price));
    checkCuda(cudaFreeHost(h_strike));
    checkCuda(cudaFreeHost(h_t));
    checkCuda(cudaFreeHost(h_rate));
    checkCuda(cudaFreeHost(h_vol));
    checkCuda(cudaFreeHost(h_call));
    checkCuda(cudaFreeHost(h_put));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  // Free device allocations
  fprintf(stderr, "Freeing device allocations...");
  fflush(stdout);
  checkCuda(cudaFree(d_price));
  checkCuda(cudaFree(d_strike));
  checkCuda(cudaFree(d_t));
  checkCuda(cudaFree(d_rate));
  checkCuda(cudaFree(d_vol));
  checkCuda(cudaFree(d_call));
  checkCuda(cudaFree(d_put));
  if (mode != CUDA_FUSED) {
    checkCuda(cudaFree(d_rsig));
    checkCuda(cudaFree(d_vol_sqrt));
    checkCuda(cudaFree(d_tmp));
    checkCuda(cudaFree(d_d1));
    checkCuda(cudaFree(d_d2));
  }
  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);
  return result;
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (mode == UNKNOWN) {
    print_usage(argv);
    exit(EXIT_FAILURE);
  }

  // Need to call this before any of the other library functions.
  if (mode == MKL_COMPOSER || mode == MKL_COMPOSER_NOPIPE) {
    composer_init(threads, piece_size);
    mkl_set_num_threads(1);
    omp_set_num_threads(1);
  } else if (mode == MKL) {
    mkl_set_num_threads(threads);
  } else {
    omp_set_num_threads(threads);
  }

  printf("Data Size: %ld Piece Size: %ld Threads: %ld Mode: %d Pinned: %d nStreams: %ld \n",
      data_size, piece_size, threads, mode, pinned, nstreams);

  struct timeval start;
  gettimeofday(&start, NULL);

  // Generate inputs.
  fprintf(stderr, "Initializing...");
  fflush(stdout);
  int lazy = (mode == MKL_COMPOSER || mode == MKL_COMPOSER_NOPIPE);
  vec_t price = vvals(data_size, 4.0, lazy);
  vec_t strike = vvals(data_size, 4.0, lazy);
  vec_t t = vvals(data_size, 4.0, lazy);
  vec_t rate = vvals(data_size, 4.0, lazy);
  vec_t vol = vvals(data_size, 4.0, lazy);

  vec_t vrsig = vvals(data_size, 0, 0);
  vec_t vvol_sqrt = vvals(data_size, 0, 0);
  vec_t vtmp = vvals(data_size, 0, 0);
  vec_t vd1 = vvals(data_size, 0, 0);
  vec_t vd2 = vvals(data_size, 0, 0);

  // Mark these as lazy.
  vec_t vcall = vvals(data_size, 0, lazy);
  vec_t vput = vvals(data_size, 0, lazy);

  fprintf(stderr, "done: %fs\n", runtime(start));
  fflush(stdout);

  fprintf(stderr, "Allocated Input Bytes: %ld\n", data_size * sizeof(double) * 5);

  fprintf(stderr, "--------------------\n");

  // Determine whether we are using cuda
  result_t result;
  switch (mode) {
    case CUDA_FUSED:
    case CUDA_FUSED_NOPIPE:
    case CUDA_STREAM:
    case CUDA_STREAM_NOPIPE:
      result = run_cuda_main(price, strike, t, rate, vol, vcall, vput);
      break;
    default:
      result = run_mkl_main(price, strike, t, rate, vol,
                            vrsig, vvol_sqrt, vtmp, vd1, vd2, vcall, vput);
  }

  for (int j = 0; j < 5; j++) {
    long index = rand() % data_size;
    fprintf(stderr, "(%f, %f) ", result.call.data[index], result.put.data[index]);
    fprintf(stdout, "(%f, %f) ", result.call.data[index], result.put.data[index]);
  }
  printf("\n");

  printf("Checking correctness...\n");
  for (int i = 0; i < data_size; i++) {
    if (result.call.data[0] != result.call.data[i]) {
      printf("Call mismatch at position %d\n", i);
      exit(1);
    }
    if (result.put.data[0] != result.put.data[i]) {
      printf("Put mismatch at position %d\n", i);
      exit(1);
    }
  }
  fprintf(stderr, "\n");
  fflush(stderr);
  fflush(stdout);
  printf("Runtime: %f s\n", result.runtime);
}
