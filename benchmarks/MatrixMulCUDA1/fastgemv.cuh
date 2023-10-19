#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <driver_functions.h>

#include <iostream>

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

void GEMV1(float *dA, float *dB, float *dC, int m, int n, int k);