#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_functions.h>

#include <iostream>

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

template <typename T>
void GEMV3(T *dVecTrans, T *dMatTrans, T *dResTrans, int m, int n, int k);
