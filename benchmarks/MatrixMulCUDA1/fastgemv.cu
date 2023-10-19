#include "MatrixMulCUDA1/fastgemv.cuh"



__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

// a = mxk, b = kxn
__global__ void fastgemv(float* mat, float* vec, float* res, unsigned int n, unsigned int k,
                          unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  unsigned int block_dim_x = blockDim.x;

  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);
  int remainder = n % (block_dim_x * 4);
  if (row >= k)
    return;

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 2; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;

    if (j < n >> 2) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 2) + j];
      sum += vec_val.x * mat_val.x;
      sum += vec_val.y * mat_val.y;
      sum += vec_val.z * mat_val.z;
      sum += vec_val.w * mat_val.w;
    }
  }

  unsigned int j = start_idx + (num_per_thread >> 2) * block_dim_x;
  if (0 != remainder)
  {
    if (j * 4 < n)
    {
      float* vec_t = (float*)&vec4[j];
      float4* mat_val = reinterpret_cast<float4*>(mat + row * n);
      float* mat_t = (float*)&mat_val[j];
      for (int i = 0; i < 4; i++)
      {
        if ( j * 4 + i < n){
          sum += vec_t[i] * mat_t[i];
        }
      }
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = sum;
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = sum;
  }
}

void GEMV1(float *dVecTrans, float *dMatTrans, float *dResTrans, int m, int n, int k) {
  int mat_height_ = n;
  int vec_height_ = k;

  // printf("n = %d  k = %d \n", n, k);

  int block_dim_x = 128;
  int block_dim_y = 4;
  // assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  // assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = vec_height_ / block_dim_x;
  // assert(num_per_thread >= 8);

  dim3 grid_dim(1, (mat_height_ + block_dim_y -1) / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);

  fastgemv<<<grid_dim, block_dim>>>(
      dMatTrans,  
      dVecTrans, 
      dResTrans,  
      vec_height_, 
      mat_height_,
      num_per_thread);
}

